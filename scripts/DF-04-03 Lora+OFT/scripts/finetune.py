# =============================================================================
# DF-04-03 LoRA+OFT Trainer
# =============================================================================
#
# The project wrapper reads the YAML configuration and passes every supported
# value to this file as an explicit command-line argument. This trainer does
# not define experiment-specific defaults for model, data, LoRA, optimization,
# validation, or checkpoint settings.
#
# The implementation keeps the official OpenVLA-OFT continuous L1 action path
# and adds fixed-split loading, train-only normalization, validation, best and
# final adapter saving, resumable checkpoints, and structured runtime records.
# =============================================================================


import ast
import hashlib
import json
import os
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # Model values passed by the YAML wrapper.
    vla_path: str
    model_hub_id: str
    local_files_only: bool
    torch_dtype: str
    device: str

    # Dataset, split, and normalization values passed by the YAML wrapper.
    data_root_dir: Path
    dataset_name: str
    episode_split_json: Path
    train_split_name: str
    val_split_name: str
    test_split_name: str
    expected_train_episodes: int
    expected_val_episodes: int
    expected_test_episodes: int
    split_seed: int
    normalization_stats_json: Path
    normalization_stats_source: str
    allow_pipeline_debug_stats: bool

    # Output paths passed by the YAML wrapper.
    run_root_dir: Path
    runtime_resolved_config_path: Path
    training_summary_path: Path
    validation_history_path: Path
    launch_resolved_config_path: Path

    # OFT values passed by the YAML wrapper.
    use_l1_regression: bool
    use_diffusion: bool
    use_film: bool
    num_images_in_input: int
    use_proprio: bool

    # Optimization values passed by the YAML wrapper.
    batch_size: int
    max_steps: int
    learning_rate: float
    lr_warmup_steps: int
    num_steps_before_decay: int
    grad_accumulation_steps: int
    image_aug: bool
    shuffle_buffer_size: int
    use_quantization: bool
    validate_finite_values: bool
    random_seed: int

    # LoRA values passed by the YAML wrapper.
    use_lora: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    init_lora_weights: str

    # Logging, validation, and checkpoint values passed by the YAML wrapper.
    console_log_interval_steps: int
    validation_start_step: int
    validation_interval_steps: int
    validation_max_batches: str
    validation_time_limit_seconds: str
    best_model_start_step: int
    overfit_patience: int
    overfit_min_delta: float
    resume_checkpoint_steps: str
    save_total_limit: int
    resume_from_checkpoint: str

    # Wrapper metadata values that are stored in runtime records.
    wandb_project: str
    run_id_note: Optional[str]


# Return the current UTC timestamp in ISO-8601 format.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Synchronize all distributed ranks when DDP is active.
def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# Return whether the current rank is responsible for output files.
def is_main_process(state: PartialState) -> bool:
    return bool(state.is_main_process)


# Return the active DDP process count.
def distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


# Create the parent directory required by an output file.
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Write one JSON object atomically.
def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    ensure_parent(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# Append one JSON object to a JSON Lines file.
def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        )
        file.flush()


# Compute a SHA256 hash without loading the whole file into memory.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Compute a deterministic SHA256 hash for a JSON-serializable value.
def sha256_json_value(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Load a JSON file and require an object at the root.
def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


# Convert an optional CLI value into a positive integer or None.
def parse_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive or none.")
    return parsed


# Convert an optional CLI path into an absolute Path or None.
def parse_optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return Path(normalized).expanduser().resolve()


# Parse the wrapper-provided resumable checkpoint step list.
def parse_step_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parsed: Any = list(value)
    else:
        text = str(value).strip()
        if text.lower() in {"", "none", "null", "[]"}:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise TypeError(
            "resume_checkpoint_steps must resolve to a list of integers."
        )
    result = [int(item) for item in parsed]
    if any(step <= 0 for step in result):
        raise ValueError("Every resume checkpoint step must be positive.")
    if result != sorted(set(result)):
        raise ValueError(
            "resume_checkpoint_steps must be sorted and contain no duplicates."
        )
    return result


# Convert the wrapper-provided LoRA target modules into PEFT format.
def parse_target_modules(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("lora_target_modules cannot be empty.")
        if normalized == "all-linear":
            return normalized
        if "," in normalized:
            modules = [item.strip() for item in normalized.split(",") if item.strip()]
            if not modules:
                raise ValueError("lora_target_modules contains no valid modules.")
            return modules
        return normalized
    if isinstance(value, (list, tuple)):
        modules = [str(item).strip() for item in value if str(item).strip()]
        if not modules:
            raise ValueError("lora_target_modules contains no valid modules.")
        return modules
    raise TypeError("lora_target_modules must be a string or sequence.")


# Resolve the wrapper-provided dtype into a Torch dtype.
def resolve_torch_dtype(value: str) -> torch.dtype:
    normalized = str(value).strip().lower()
    supported = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in supported:
        raise ValueError(
            "torch_dtype must be one of: bfloat16, bf16, float16, fp16."
        )
    return supported[normalized]


# Validate the fixed split file against wrapper-provided counts and seed.
def validate_split_manifest(
    split_path: Path,
    cfg: FinetuneConfig,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    document = load_json_object(split_path)

    if document.get("random_seed") != cfg.split_seed:
        raise ValueError(
            "Split random_seed mismatch: "
            f"actual={document.get('random_seed')!r}, expected={cfg.split_seed}."
        )
    if document.get("source_split") != "train":
        raise ValueError("The physical RLDS source split must be 'train'.")

    totals = document.get("totals")
    if not isinstance(totals, dict):
        raise KeyError("Split JSON is missing totals.")

    expected_original = (
        cfg.expected_train_episodes
        + cfg.expected_val_episodes
        + cfg.expected_test_episodes
    )
    expected_totals = {
        "original_episode_count": expected_original,
        "train_count": cfg.expected_train_episodes,
        "val_count": cfg.expected_val_episodes,
        "test_count": cfg.expected_test_episodes,
    }
    for field_name, expected_value in expected_totals.items():
        if totals.get(field_name) != expected_value:
            raise ValueError(
                f"Split totals.{field_name} mismatch: "
                f"actual={totals.get(field_name)!r}, expected={expected_value}."
            )

    records = document.get("sub_datasets")
    if not isinstance(records, list) or not records:
        raise KeyError("Split JSON is missing sub_datasets.")

    by_name: Dict[str, Dict[str, Any]] = {}
    verified_totals = {
        "original_episode_count": 0,
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
    }

    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Every split sub_datasets record must be an object.")
        dataset_name = record.get("sub_dataset")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise TypeError("Every split record requires a sub_dataset string.")
        if dataset_name in by_name:
            raise ValueError(f"Duplicate split record: {dataset_name}")
        if record.get("source_split") != document.get("source_split"):
            raise ValueError(f"{dataset_name}.source_split is inconsistent.")

        original_count = int(record["original_episode_count"])
        train_count = int(record["train_count"])
        val_count = int(record["val_count"])
        test_count = int(record["test_count"])

        split_indices: Dict[str, List[int]] = {}
        for split_name, declared_count in (
            ("train", train_count),
            ("val", val_count),
            ("test", test_count),
        ):
            key = f"{split_name}_episode_indices"
            indices = record.get(key)
            if not isinstance(indices, list):
                raise TypeError(f"{dataset_name}.{key} must be a list.")
            if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
                raise TypeError(f"{dataset_name}.{key} must contain integers.")
            if len(indices) != declared_count:
                raise ValueError(
                    f"{dataset_name}.{key} count mismatch: "
                    f"indices={len(indices)}, declared={declared_count}."
                )
            if len(indices) != len(set(indices)):
                raise ValueError(f"{dataset_name}.{key} contains duplicates.")
            if any(index < 0 or index >= original_count for index in indices):
                raise ValueError(f"{dataset_name}.{key} contains an invalid index.")
            split_indices[split_name] = indices

        all_indices = (
            split_indices["train"]
            + split_indices["val"]
            + split_indices["test"]
        )
        if len(all_indices) != original_count or set(all_indices) != set(range(original_count)):
            raise ValueError(
                f"{dataset_name} train/val/test indices are not a complete "
                "disjoint partition."
            )

        by_name[dataset_name] = record
        verified_totals["original_episode_count"] += original_count
        verified_totals["train_count"] += train_count
        verified_totals["val_count"] += val_count
        verified_totals["test_count"] += test_count

    if verified_totals != expected_totals:
        raise ValueError(
            f"Per-dataset split totals mismatch: {verified_totals}."
        )

    return document, by_name


# Validate one action or proprio statistics section.
def validate_statistics_section(
    dataset_name: str,
    section_name: str,
    section: Any,
    expected_dim: int,
) -> None:
    if not isinstance(section, dict):
        raise TypeError(f"{dataset_name}.{section_name} must be an object.")

    numeric_fields = ("mean", "std", "min", "max", "q01", "q99")
    for field_name in (*numeric_fields, "mask"):
        values = section.get(field_name)
        if not isinstance(values, list) or len(values) != expected_dim:
            raise ValueError(
                f"{dataset_name}.{section_name}.{field_name} must contain "
                f"{expected_dim} values."
            )

    arrays: Dict[str, np.ndarray] = {}
    for field_name in numeric_fields:
        values = np.asarray(section[field_name], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{dataset_name}.{section_name}.{field_name} contains NaN or Inf."
            )
        arrays[field_name] = values

    if np.any(arrays["std"] < 0.0):
        raise ValueError(f"{dataset_name}.{section_name}.std is negative.")
    if not np.all(
        (arrays["min"] <= arrays["q01"])
        & (arrays["q01"] <= arrays["q99"])
        & (arrays["q99"] <= arrays["max"])
    ):
        raise ValueError(
            f"{dataset_name}.{section_name} has invalid min/q01/q99/max ordering."
        )
    if any(not isinstance(value, bool) for value in section["mask"]):
        raise TypeError(f"{dataset_name}.{section_name}.mask must be boolean.")


# Load train-only normalization statistics and verify split provenance.
def load_normalization_statistics(
    path: Path,
    split_path: Path,
    split_records: Mapping[str, Mapping[str, Any]],
    cfg: FinetuneConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    document = load_json_object(path)
    metadata = document.get("metadata")
    datasets = document.get("datasets")

    if not isinstance(metadata, dict):
        raise KeyError("Normalization statistics JSON is missing metadata.")
    if not isinstance(datasets, dict):
        raise KeyError("Normalization statistics JSON is missing datasets.")

    pipeline_debug_only = metadata.get("pipeline_debug_only")
    if not isinstance(pipeline_debug_only, bool):
        raise TypeError("metadata.pipeline_debug_only must be boolean.")
    if pipeline_debug_only and not cfg.allow_pipeline_debug_stats:
        raise ValueError("Pipeline-debug statistics are not allowed.")

    expected_metadata = {
        "stats_source": cfg.normalization_stats_source,
        "computed_from_split": cfg.train_split_name,
        "episode_count": cfg.expected_train_episodes,
        "split_seed": cfg.split_seed,
        "physical_source_split": "train",
        "action_dim": ACTION_DIM,
        "proprio_dim": PROPRIO_DIM,
    }
    for field_name, expected_value in expected_metadata.items():
        if metadata.get(field_name) != expected_value:
            raise ValueError(
                f"Normalization metadata.{field_name} mismatch: "
                f"actual={metadata.get(field_name)!r}, expected={expected_value!r}."
            )

    split_sha256 = sha256_file(split_path)
    if metadata.get("split_json_sha256") != split_sha256:
        raise ValueError("Normalization split_json_sha256 does not match the split file.")

    split_dataset_names = list(split_records.keys())
    if set(datasets) != set(split_dataset_names):
        raise ValueError(
            "Normalization dataset keys do not match the split dataset keys."
        )
    metadata_dataset_names = metadata.get("dataset_names")
    if metadata_dataset_names is not None and set(metadata_dataset_names) != set(split_dataset_names):
        raise ValueError("Normalization metadata.dataset_names is inconsistent.")

    trajectory_total = 0
    transition_total = 0
    for dataset_name in split_dataset_names:
        stats = datasets[dataset_name]
        if not isinstance(stats, dict):
            raise TypeError(f"Statistics for {dataset_name} must be an object.")

        expected_trajectories = int(split_records[dataset_name]["train_count"])
        trajectories = int(stats.get("num_trajectories", -1))
        transitions = int(stats.get("num_transitions", -1))
        if trajectories != expected_trajectories:
            raise ValueError(
                f"{dataset_name}.num_trajectories mismatch: "
                f"actual={trajectories}, expected={expected_trajectories}."
            )
        if transitions <= 0:
            raise ValueError(f"{dataset_name}.num_transitions must be positive.")
        if stats.get("source_split") != cfg.train_split_name:
            raise ValueError(f"{dataset_name}.source_split mismatch.")

        expected_indices_hash = sha256_json_value(
            split_records[dataset_name]["train_episode_indices"]
        )
        if stats.get("selected_episode_indices_sha256") != expected_indices_hash:
            raise ValueError(
                f"{dataset_name}.selected_episode_indices_sha256 mismatch."
            )

        validate_statistics_section(
            dataset_name,
            "action",
            stats.get("action"),
            ACTION_DIM,
        )
        validate_statistics_section(
            dataset_name,
            "proprio",
            stats.get("proprio"),
            PROPRIO_DIM,
        )

        trajectory_total += trajectories
        transition_total += transitions

    if trajectory_total != cfg.expected_train_episodes:
        raise ValueError(
            "Normalization trajectory total does not match expected_train_episodes."
        )
    if metadata.get("transition_count") != transition_total:
        raise ValueError(
            "Normalization metadata.transition_count does not match dataset totals."
        )

    return document, metadata, datasets


# Reject incompatible wrapper-provided training values before GPU allocation.
def validate_config(cfg: FinetuneConfig) -> None:
    model_path = Path(cfg.vla_path).expanduser()
    dataset_root = Path(cfg.data_root_dir).expanduser()
    split_path = Path(cfg.episode_split_json).expanduser()
    stats_path = Path(cfg.normalization_stats_json).expanduser()

    if not cfg.local_files_only:
        raise ValueError("DF-04-03 requires local_files_only=true.")
    resolve_torch_dtype(cfg.torch_dtype)
    if cfg.device != "cuda":
        raise ValueError("DF-04-03 requires device='cuda'.")
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid local model snapshot: {model_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Split JSON does not exist: {split_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Statistics JSON does not exist: {stats_path}")

    if any(
        value <= 0
        for value in (
            cfg.expected_train_episodes,
            cfg.expected_val_episodes,
            cfg.expected_test_episodes,
        )
    ):
        raise ValueError("Expected episode counts must be positive.")
    if cfg.split_seed < 0:
        raise ValueError("split_seed cannot be negative.")
    if not cfg.normalization_stats_source.strip():
        raise ValueError("normalization_stats_source cannot be empty.")

    if not cfg.use_lora:
        raise ValueError("DF-04-03 requires use_lora=true.")
    if not cfg.use_l1_regression:
        raise ValueError("DF-04-03 requires use_l1_regression=true.")
    if cfg.use_diffusion:
        raise ValueError("DF-04-03 does not use diffusion.")
    if cfg.num_images_in_input <= 0:
        raise ValueError("num_images_in_input must be positive.")
    if not cfg.use_proprio:
        raise ValueError("DF-04-03 requires use_proprio=true.")
    if cfg.use_quantization:
        raise ValueError("DF-04-03 is not a quantized-training run.")

    if cfg.train_split_name != "train":
        raise ValueError("train_split_name must be 'train'.")
    if cfg.val_split_name != "val":
        raise ValueError("val_split_name must be 'val'.")
    if cfg.test_split_name != "test":
        raise ValueError("test_split_name must be 'test'.")

    if cfg.batch_size <= 0 or cfg.max_steps <= 0:
        raise ValueError("batch_size and max_steps must be positive.")
    if cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if cfg.grad_accumulation_steps <= 0:
        raise ValueError("grad_accumulation_steps must be positive.")
    if cfg.shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be positive.")
    if cfg.console_log_interval_steps <= 0:
        raise ValueError("console_log_interval_steps must be positive.")
    if cfg.lr_warmup_steps < 0 or cfg.lr_warmup_steps > cfg.max_steps:
        raise ValueError("lr_warmup_steps must be in [0, max_steps].")
    if (
        cfg.num_steps_before_decay <= 0
        or cfg.num_steps_before_decay > cfg.max_steps
    ):
        raise ValueError("num_steps_before_decay must be in (0, max_steps].")
    if cfg.lr_warmup_steps > cfg.num_steps_before_decay:
        raise ValueError("lr_warmup_steps cannot exceed num_steps_before_decay.")

    if cfg.lora_rank <= 0 or cfg.lora_alpha <= 0:
        raise ValueError("lora_rank and lora_alpha must be positive.")
    if cfg.lora_dropout < 0.0 or cfg.lora_dropout >= 1.0:
        raise ValueError("lora_dropout must be in [0, 1).")
    parse_target_modules(cfg.lora_target_modules)
    if not str(cfg.init_lora_weights).strip():
        raise ValueError("init_lora_weights cannot be empty.")

    if cfg.validation_start_step < 0 or cfg.validation_start_step > cfg.max_steps:
        raise ValueError("validation_start_step must be in [0, max_steps].")
    if cfg.validation_interval_steps <= 0:
        raise ValueError("validation_interval_steps must be positive.")
    if cfg.best_model_start_step < 0 or cfg.best_model_start_step > cfg.max_steps:
        raise ValueError("best_model_start_step must be in [0, max_steps].")
    if cfg.overfit_patience < 2:
        raise ValueError("overfit_patience must be at least two.")
    if cfg.overfit_min_delta < 0.0:
        raise ValueError("overfit_min_delta cannot be negative.")

    parse_optional_int(cfg.validation_max_batches, "validation_max_batches")
    parse_optional_int(
        cfg.validation_time_limit_seconds,
        "validation_time_limit_seconds",
    )

    checkpoint_steps = parse_step_list(cfg.resume_checkpoint_steps)
    if any(step > cfg.max_steps for step in checkpoint_steps):
        raise ValueError("A resume checkpoint step exceeds max_steps.")
    if cfg.save_total_limit < 0:
        raise ValueError("save_total_limit cannot be negative.")
    if not str(cfg.wandb_project).strip():
        raise ValueError("wandb_project cannot be empty.")


# Register the local OpenVLA classes with Hugging Face AutoClasses.
def register_openvla_auto_classes() -> None:
    registrations = (
        (AutoConfig, ("openvla", OpenVLAConfig)),
        (AutoImageProcessor, (OpenVLAConfig, PrismaticImageProcessor)),
        (AutoProcessor, (OpenVLAConfig, PrismaticProcessor)),
        (
            AutoModelForVision2Seq,
            (OpenVLAConfig, OpenVLAForActionPrediction),
        ),
    )
    for auto_class, arguments in registrations:
        try:
            auto_class.register(*arguments)
        except ValueError:
            pass


# Seed Python, NumPy, CPU Torch, and CUDA Torch generators.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Return the underlying module when a component is wrapped by DDP.
def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


# Wrap one trainable module with DistributedDataParallel.
def wrap_ddp(
    module: nn.Module,
    device_id: int,
    find_unused_parameters: bool,
) -> DDP:
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
    )


# Count total and trainable parameters from one runtime module.
def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


# Read actual LoRA values from the constructed PEFT runtime object.
def actual_lora_config(vla: nn.Module) -> Dict[str, Any]:
    model = unwrap_ddp(vla)
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict) or not peft_config:
        raise RuntimeError("Actual PEFT configuration is unavailable.")

    active_adapter = getattr(model, "active_adapter", None)
    if isinstance(active_adapter, (list, tuple)):
        active_adapter = active_adapter[0] if active_adapter else None
    if active_adapter not in peft_config:
        active_adapter = next(iter(peft_config))

    actual = peft_config[active_adapter]
    target_modules = actual.target_modules
    if isinstance(target_modules, set):
        target_modules = sorted(target_modules)
    task_type = actual.task_type
    if hasattr(task_type, "value"):
        task_type = task_type.value

    return {
        "adapter_name": active_adapter,
        "r": int(actual.r),
        "lora_alpha": int(actual.lora_alpha),
        "lora_dropout": float(actual.lora_dropout),
        "target_modules": target_modules,
        "init_lora_weights": actual.init_lora_weights,
        "bias": actual.bias,
        "task_type": str(task_type),
    }


# Attach train-only normalization statistics to the model configuration.
def set_runtime_norm_stats(
    vla_model: nn.Module,
    normalization_datasets: Mapping[str, Any],
) -> None:
    vla_model.config.norm_stats = dict(normalization_datasets)
    get_base_model = getattr(vla_model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if hasattr(base_model, "config"):
            base_model.config.norm_stats = dict(normalization_datasets)


# Move restored optimizer tensors to the active CUDA device.
def optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


# Compute the global L2 norm with only one GPU-to-CPU synchronization.
def calculate_gradient_norm(
    modules: Sequence[nn.Module],
) -> Optional[float]:
    gradient_norms: List[torch.Tensor] = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            gradient_norms.append(
                parameter.grad.detach().float().norm(2)
            )

    if not gradient_norms:
        return None

    total_norm = torch.linalg.vector_norm(
        torch.stack(gradient_norms),
        ord=2,
    )
    return float(total_norm.item())


# Convert detached scalar metric tensors to Python floats with one synchronization.
def scalar_metrics_to_cpu(
    metrics: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    metric_names = list(metrics)
    if not metric_names:
        return {}

    packed = torch.stack(
        [
            metrics[name].detach().float()
            for name in metric_names
        ]
    )
    values = packed.cpu().tolist()
    return {
        name: float(values[index])
        for index, name in enumerate(metric_names)
    }


# Run the official OpenVLA-OFT hidden-state and continuous L1 action path.
def official_l1_forward_pass(
    vla: DDP,
    action_head: DDP,
    proprio_projector: DDP,
    batch: Mapping[str, Any],
    device: torch.device,
    runtime_dtype: torch.dtype,
    num_patches: int,
    use_film: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ground_truth_actions = batch["actions"].to(
        device=device,
        dtype=runtime_dtype,
        non_blocking=True,
    )
    labels = batch["labels"].to(device=device, non_blocking=True)
    proprio = batch["proprio"].to(
        device=device,
        dtype=runtime_dtype,
        non_blocking=True,
    )

    expected_action_shape = (
        ground_truth_actions.shape[0],
        NUM_ACTIONS_CHUNK,
        ACTION_DIM,
    )
    if tuple(ground_truth_actions.shape) != expected_action_shape:
        raise ValueError(
            "Ground-truth action shape mismatch: "
            f"actual={tuple(ground_truth_actions.shape)}, "
            f"expected={expected_action_shape}."
        )
    expected_proprio_shape = (
        ground_truth_actions.shape[0],
        PROPRIO_DIM,
    )
    if tuple(proprio.shape) != expected_proprio_shape:
        raise ValueError(
            "Proprio shape mismatch: "
            f"actual={tuple(proprio.shape)}, expected={expected_proprio_shape}."
        )

    with torch.autocast(device_type="cuda", dtype=runtime_dtype):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device=device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(
                device=device,
                non_blocking=True,
            ),
            pixel_values=batch["pixel_values"].to(
                device=device,
                dtype=runtime_dtype,
                non_blocking=True,
            ),
            labels=labels,
            output_hidden_states=True,
            proprio=proprio,
            proprio_projector=proprio_projector,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
        )

        ground_truth_token_ids = labels[:, 1:]
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        action_mask = current_action_mask | next_actions_mask
        selected_action_hidden_states = text_hidden_states[action_mask]

        batch_size = ground_truth_actions.shape[0]
        expected_action_tokens = batch_size * NUM_ACTIONS_CHUNK * ACTION_DIM
        if selected_action_hidden_states.shape[0] != expected_action_tokens:
            raise RuntimeError(
                "Selected action hidden-state count mismatch: "
                f"actual={selected_action_hidden_states.shape[0]}, "
                f"expected={expected_action_tokens}."
            )

        actions_hidden_states = selected_action_hidden_states.reshape(
            batch_size,
            NUM_ACTIONS_CHUNK * ACTION_DIM,
            -1,
        ).to(runtime_dtype)
        predicted_actions = unwrap_ddp(action_head).predict_action(
            actions_hidden_states
        )

        if tuple(predicted_actions.shape) != tuple(ground_truth_actions.shape):
            raise RuntimeError(
                "Predicted action shape mismatch: "
                f"actual={tuple(predicted_actions.shape)}, "
                f"expected={tuple(ground_truth_actions.shape)}."
            )

        loss = nn.functional.l1_loss(
            predicted_actions,
            ground_truth_actions,
            reduction="mean",
        )
        current_action_l1 = nn.functional.l1_loss(
            predicted_actions[:, 0],
            ground_truth_actions[:, 0],
            reduction="mean",
        )
        future_actions_l1 = nn.functional.l1_loss(
            predicted_actions[:, 1:],
            ground_truth_actions[:, 1:],
            reduction="mean",
        )

    return loss, {
        "loss_value": loss.detach().float(),
        "curr_action_l1_loss": current_action_l1.detach().float(),
        "next_actions_l1_loss": future_actions_l1.detach().float(),
    }


# Return whether validation is scheduled at the current optimization step.
def should_run_validation(
    global_step: int,
    start_step: int,
    interval_steps: int,
) -> bool:
    if global_step < start_step:
        return False
    return (global_step - start_step) % interval_steps == 0


# Detect a rising-validation and falling-training loss pattern.
def analyse_overfitting(
    history: Sequence[Mapping[str, Any]],
    patience: int,
    min_delta: float,
) -> Dict[str, Any]:
    points = [
        row
        for row in history
        if row.get("train_loss") is not None and row.get("val_loss") is not None
    ]
    result: Dict[str, Any] = {
        "overfitting_checked": len(points) >= patience,
        "overfitting_detected": False,
        "overfitting_reason": "insufficient_validation_points",
        "overfitting_points_used": min(len(points), patience),
    }
    if len(points) < patience:
        return result

    recent = points[-patience:]
    previous = points[:-patience]
    previous_best = (
        min(float(row["val_loss"]) for row in previous)
        if previous
        else float(recent[0]["val_loss"])
    )
    recent_val_losses = [float(row["val_loss"]) for row in recent]
    recent_train_losses = [float(row["train_loss"]) for row in recent]

    detected = (
        min(recent_val_losses) >= previous_best - min_delta
        and recent_val_losses[-1] > recent_val_losses[0] + min_delta
        and recent_train_losses[-1] < recent_train_losses[0]
    )
    result.update(
        {
            "overfitting_detected": bool(detected),
            "overfitting_reason": (
                "validation_loss_rising_while_train_loss_falling"
                if detected
                else "no_clear_overfitting_signal"
            ),
            "previous_best_val_loss": previous_best,
            "recent_val_losses": recent_val_losses,
            "recent_train_losses": recent_train_losses,
        }
    )
    return result


# Evaluate validation batches and average metrics across distributed ranks.
def run_validation(
    vla: DDP,
    action_head: DDP,
    proprio_projector: DDP,
    val_dataloader: DataLoader,
    device: torch.device,
    runtime_dtype: torch.dtype,
    num_patches: int,
    use_film: bool,
    max_batches: Optional[int],
    time_limit_seconds: Optional[int],
) -> Dict[str, Any]:
    vla.eval()
    action_head.eval()
    proprio_projector.eval()

    metric_sums: Dict[str, torch.Tensor] = {}
    local_batch_count = 0
    started_at = time.time()
    stop_reason = "dataloader_exhausted"

    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = official_l1_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device=device,
                runtime_dtype=runtime_dtype,
                num_patches=num_patches,
                use_film=use_film,
            )
            for name, value in metrics.items():
                metric_value = value.detach().to(
                    device=device,
                    dtype=torch.float64,
                )
                if name not in metric_sums:
                    metric_sums[name] = metric_value
                else:
                    metric_sums[name].add_(metric_value)
            local_batch_count += 1

            if max_batches is not None and local_batch_count >= max_batches:
                stop_reason = "max_batches"
                break
            if time_limit_seconds is not None:
                torch.cuda.synchronize(device)
                if time.time() - started_at >= time_limit_seconds:
                    stop_reason = "time_limit"
                    break

    metric_names = sorted(metric_sums)
    packed = torch.stack(
        [metric_sums[name] for name in metric_names]
        + [
            torch.tensor(
                float(local_batch_count),
                device=device,
                dtype=torch.float64,
            )
        ]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    packed_values = packed.cpu().tolist()
    global_batch_count = int(packed_values[-1])
    if global_batch_count <= 0:
        raise RuntimeError("Validation produced zero batches.")

    averaged = {
        name: float(packed_values[index]) / float(global_batch_count)
        for index, name in enumerate(metric_names)
    }
    averaged["val_loss"] = averaged["loss_value"]
    averaged["val_batches_count"] = global_batch_count
    averaged["validation_time_seconds"] = round(
        time.time() - started_at,
        6,
    )
    averaged["validation_stop_reason"] = stop_reason
    averaged["validation_completed"] = stop_reason == "dataloader_exhausted"

    vla.train()
    action_head.train()
    proprio_projector.train()
    return averaged


# Save one unwrapped trainable component state dictionary.
def save_component_state(module: nn.Module, output_path: Path) -> None:
    torch.save(unwrap_ddp(module).state_dict(), output_path)


# Atomically save the adapter, OFT heads, processor, and statistics.
def save_adapter_bundle(
    output_dir: Path,
    vla: DDP,
    action_head: DDP,
    proprio_projector: DDP,
    processor: Any,
    compatibility_statistics: Mapping[str, Any],
    provenance_statistics: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]],
) -> None:
    temporary_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=True)

    unwrap_ddp(vla).save_pretrained(temporary_dir, safe_serialization=True)
    save_component_state(action_head, temporary_dir / "action_head.pt")
    save_component_state(
        proprio_projector,
        temporary_dir / "proprio_projector.pt",
    )
    processor.save_pretrained(temporary_dir / "processor")
    write_json_atomic(
        temporary_dir / "dataset_statistics.json",
        compatibility_statistics,
    )
    write_json_atomic(
        temporary_dir / "dataset_statistics_provenance.json",
        provenance_statistics,
    )
    if metrics is not None:
        write_json_atomic(
            temporary_dir / "validation_metrics.json",
            dict(metrics),
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(temporary_dir, output_dir)


# Capture Python, NumPy, CPU Torch, and CUDA RNG states.
def capture_rng_state() -> Dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


# Restore all RNG states stored in a resumable checkpoint.
def restore_rng_state(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state.get("torch_cuda_rng_state_all", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


# Save adapter components, optimizer, scheduler, RNG, and trainer state.
def save_resume_checkpoint(
    checkpoint_dir: Path,
    global_step: int,
    vla: DDP,
    action_head: DDP,
    proprio_projector: DDP,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: MultiStepLR,
    trainer_state: Mapping[str, Any],
    compatibility_statistics: Mapping[str, Any],
    provenance_statistics: Mapping[str, Any],
) -> None:
    temporary_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=True)

    unwrap_ddp(vla).save_pretrained(temporary_dir, safe_serialization=True)
    save_component_state(action_head, temporary_dir / "action_head.pt")
    save_component_state(
        proprio_projector,
        temporary_dir / "proprio_projector.pt",
    )
    torch.save(optimizer.state_dict(), temporary_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary_dir / "scheduler.pt")
    torch.save(capture_rng_state(), temporary_dir / "rng_state.pt")
    processor.save_pretrained(temporary_dir / "processor")
    write_json_atomic(
        temporary_dir / "dataset_statistics.json",
        compatibility_statistics,
    )
    write_json_atomic(
        temporary_dir / "dataset_statistics_provenance.json",
        provenance_statistics,
    )

    state_document = dict(trainer_state)
    state_document["global_step"] = global_step
    state_document["dataloader_stream_exact_resume"] = False
    state_document["dataloader_resume_note"] = (
        "Model, optimizer, scheduler, and RNG states are restored. "
        "The repeating RLDS stream position is not restored exactly."
    )
    write_json_atomic(
        temporary_dir / "trainer_state.json",
        state_document,
    )

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    os.replace(temporary_dir, checkpoint_dir)


# Load one action-head or proprio-projector state dictionary.
def load_module_state(
    module: nn.Module,
    checkpoint_dir: Path,
    filename: str,
) -> None:
    path = checkpoint_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint component does not exist: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state_dict)


# Delete the oldest resumable checkpoints when a limit is configured.
def prune_resume_checkpoints(run_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints: List[Tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0])
    while len(checkpoints) > save_total_limit:
        _, old_path = checkpoints.pop(0)
        shutil.rmtree(old_path)


# Build runtime metadata from constructed objects and loaded files.
def build_runtime_resolved_config(
    cfg: FinetuneConfig,
    vla: DDP,
    action_head: DDP,
    proprio_projector: DDP,
    model_path: Path,
    split_path: Path,
    split_document: Mapping[str, Any],
    stats_path: Path,
    stats_metadata: Mapping[str, Any],
    stats_datasets: Mapping[str, Any],
    device: torch.device,
    validation_max_batches: Optional[int],
    validation_time_limit_seconds: Optional[int],
    checkpoint_steps: Sequence[int],
    initial_global_step: int,
    launch_resolved_config_path: Path,
) -> Dict[str, Any]:
    vla_total, vla_trainable = count_parameters(unwrap_ddp(vla))
    action_total, action_trainable = count_parameters(unwrap_ddp(action_head))
    proprio_total, proprio_trainable = count_parameters(
        unwrap_ddp(proprio_projector)
    )

    return {
        "created_at": utc_now(),
        "run_id_note": cfg.run_id_note,
        "random_seed": cfg.random_seed,
        "tracking": {
            "wandb_project": cfg.wandb_project,
            "wandb_logging_enabled": False,
        },
        "launch_config": {
            "path": str(launch_resolved_config_path),
            "sha256": (
                sha256_file(launch_resolved_config_path)
                if launch_resolved_config_path.is_file()
                else None
            ),
        },
        "model": {
            "hub_id": cfg.model_hub_id,
            "loaded_from": str(model_path),
            "local_files_only": cfg.local_files_only,
            "requested_torch_dtype": cfg.torch_dtype,
            "requested_device": cfg.device,
            "torch_dtype": str(next(unwrap_ddp(vla).parameters()).dtype),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(
                device.index if device.index is not None else 0
            ),
        },
        "dataset": {
            "name": cfg.dataset_name,
            "data_root_dir": str(Path(cfg.data_root_dir).expanduser().resolve()),
            "episode_split_json": str(split_path),
            "episode_split_sha256": sha256_file(split_path),
            "split_seed": split_document.get("random_seed"),
            "split_totals": split_document.get("totals"),
            "train_split_name": cfg.train_split_name,
            "val_split_name": cfg.val_split_name,
            "test_split_name": cfg.test_split_name,
            "normalization_stats_json": str(stats_path),
            "normalization_stats_sha256": sha256_file(stats_path),
            "normalization_stats_source": stats_metadata.get("stats_source"),
            "normalization_dataset_keys": sorted(stats_datasets.keys()),
        },
        "oft": {
            "use_l1_regression": cfg.use_l1_regression,
            "use_diffusion": cfg.use_diffusion,
            "use_film": cfg.use_film,
            "num_images_in_input": cfg.num_images_in_input,
            "use_proprio": cfg.use_proprio,
            "action_dim": ACTION_DIM,
            "proprio_dim": PROPRIO_DIM,
            "num_actions_chunk": NUM_ACTIONS_CHUNK,
            "normalization_type": str(ACTION_PROPRIO_NORMALIZATION_TYPE),
        },
        "lora_config": actual_lora_config(vla),
        "training": {
            "per_device_batch_size": cfg.batch_size,
            "world_size": distributed_world_size(),
            "grad_accumulation_steps": cfg.grad_accumulation_steps,
            "effective_global_batch_size": (
                cfg.batch_size
                * distributed_world_size()
                * cfg.grad_accumulation_steps
            ),
            "learning_rate": cfg.learning_rate,
            "lr_warmup_steps": cfg.lr_warmup_steps,
            "num_steps_before_decay": cfg.num_steps_before_decay,
            "max_steps": cfg.max_steps,
            "image_aug": cfg.image_aug,
            "shuffle_buffer_size": cfg.shuffle_buffer_size,
            "validate_finite_values": cfg.validate_finite_values,
            "optimizer": "AdamW",
            "scheduler": "MultiStepLR",
            "initial_global_step": initial_global_step,
        },
        "validation": {
            "validation_start_step": cfg.validation_start_step,
            "validation_interval_steps": cfg.validation_interval_steps,
            "validation_max_batches": validation_max_batches,
            "validation_time_limit_seconds": validation_time_limit_seconds,
            "best_model_start_step": cfg.best_model_start_step,
            "overfit_patience": cfg.overfit_patience,
            "overfit_min_delta": cfg.overfit_min_delta,
        },
        "checkpointing": {
            "resume_checkpoint_steps": list(checkpoint_steps),
            "save_total_limit": cfg.save_total_limit,
            "resume_from_checkpoint": cfg.resume_from_checkpoint,
        },
        "parameters": {
            "vla_total_params": vla_total,
            "vla_trainable_params": vla_trainable,
            "action_head_total_params": action_total,
            "action_head_trainable_params": action_trainable,
            "proprio_projector_total_params": proprio_total,
            "proprio_projector_trainable_params": proprio_trainable,
            "total_params": vla_total + action_total + proprio_total,
            "trainable_params": (
                vla_trainable + action_trainable + proprio_trainable
            ),
        },
        "software": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }


# Orchestrate model loading, datasets, training, validation, and saving.
@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    validate_config(cfg)

    model_path = Path(cfg.vla_path).expanduser().resolve()
    dataset_root = Path(cfg.data_root_dir).expanduser().resolve()
    split_path = Path(cfg.episode_split_json).expanduser().resolve()
    stats_path = Path(cfg.normalization_stats_json).expanduser().resolve()
    run_dir = Path(cfg.run_root_dir).expanduser().resolve()
    runtime_config_path = Path(
        cfg.runtime_resolved_config_path
    ).expanduser().resolve()
    training_summary_path = Path(cfg.training_summary_path).expanduser().resolve()
    validation_history_path = Path(
        cfg.validation_history_path
    ).expanduser().resolve()
    launch_resolved_config_path = Path(
        cfg.launch_resolved_config_path
    ).expanduser().resolve()

    runtime_dtype = resolve_torch_dtype(cfg.torch_dtype)
    validation_max_batches = parse_optional_int(
        cfg.validation_max_batches,
        "validation_max_batches",
    )
    validation_time_limit_seconds = parse_optional_int(
        cfg.validation_time_limit_seconds,
        "validation_time_limit_seconds",
    )
    checkpoint_steps = parse_step_list(cfg.resume_checkpoint_steps)
    resume_checkpoint = parse_optional_path(cfg.resume_from_checkpoint)

    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(runtime_config_path)
    ensure_parent(training_summary_path)
    ensure_parent(validation_history_path)
    ensure_parent(launch_resolved_config_path)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["OPENVLA_EPISODE_SPLIT_JSON"] = str(split_path)

    split_document, split_records = validate_split_manifest(split_path, cfg)
    (
        provenance_statistics,
        stats_metadata,
        compatibility_statistics,
    ) = load_normalization_statistics(
        path=stats_path,
        split_path=split_path,
        split_records=split_records,
        cfg=cfg,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("DF-04-03 training requires a CUDA GPU.")

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    device = torch.device(f"{cfg.device}:{device_id}")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_id)

    seed_everything(cfg.random_seed)
    register_openvla_auto_classes()

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=cfg.local_files_only,
    )
    base_vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=runtime_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=cfg.local_files_only,
    ).to(device)

    base_vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    llm_dim = int(base_vla.llm_dim)

    if resume_checkpoint is not None:
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(
                f"resume_from_checkpoint is not a directory: {resume_checkpoint}"
            )
        vla_model = PeftModel.from_pretrained(
            base_vla,
            resume_checkpoint,
            is_trainable=True,
            local_files_only=cfg.local_files_only,
        )
    else:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=parse_target_modules(cfg.lora_target_modules),
            init_lora_weights=cfg.init_lora_weights,
        )
        vla_model = get_peft_model(base_vla, lora_config)

    set_runtime_norm_stats(vla_model, compatibility_statistics)
    vla_model.print_trainable_parameters()

    action_head_model = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=ACTION_DIM,
    ).to(device=device, dtype=runtime_dtype)
    proprio_projector_model = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=PROPRIO_DIM,
    ).to(device=device)

    if resume_checkpoint is not None:
        load_module_state(action_head_model, resume_checkpoint, "action_head.pt")
        load_module_state(
            proprio_projector_model,
            resume_checkpoint,
            "proprio_projector.pt",
        )

    vla = wrap_ddp(vla_model, device_id, find_unused_parameters=True)
    action_head = wrap_ddp(
        action_head_model,
        device_id,
        find_unused_parameters=False,
    )
    proprio_projector = wrap_ddp(
        proprio_projector_model,
        device_id,
        find_unused_parameters=False,
    )

    num_patches = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
        + 1
    )

    trainable_parameters = [
        parameter
        for module in (vla, action_head, proprio_projector)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(trainable_parameters, lr=cfg.learning_rate)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
    )

    initial_global_step = 0
    best_val_loss: Optional[float] = None
    best_checkpoint_step: Optional[int] = None
    train_loss_curve: List[Dict[str, Any]] = []
    val_loss_curve: List[Dict[str, Any]] = []
    learning_rate_curve: List[Dict[str, Any]] = []
    gradient_norm_curve: List[Dict[str, Any]] = []
    validation_history: List[Dict[str, Any]] = []
    resume_checkpoints: List[str] = []

    if resume_checkpoint is not None:
        optimizer_path = resume_checkpoint / "optimizer.pt"
        scheduler_path = resume_checkpoint / "scheduler.pt"
        trainer_state_path = resume_checkpoint / "trainer_state.json"
        rng_state_path = resume_checkpoint / "rng_state.pt"
        for required_path in (
            optimizer_path,
            scheduler_path,
            trainer_state_path,
            rng_state_path,
        ):
            if not required_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint is missing: {required_path}")

        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(
            torch.load(scheduler_path, map_location="cpu", weights_only=True)
        )

        trainer_state = load_json_object(trainer_state_path)
        initial_global_step = int(trainer_state.get("global_step", 0))
        best_val_loss = trainer_state.get("best_val_loss")
        best_checkpoint_step = trainer_state.get("best_checkpoint_step")
        train_loss_curve = list(trainer_state.get("train_loss_curve", []))
        val_loss_curve = list(trainer_state.get("val_loss_curve", []))
        learning_rate_curve = list(
            trainer_state.get("learning_rate_curve", [])
        )
        gradient_norm_curve = list(
            trainer_state.get("gradient_norm_curve", [])
        )
        validation_history = list(
            trainer_state.get("validation_history", [])
        )
        resume_checkpoints = list(
            trainer_state.get("resume_checkpoints", [])
        )
        if initial_global_step < 0 or initial_global_step > cfg.max_steps:
            raise ValueError("Resume global_step is outside [0, max_steps].")
        restore_rng_state(rng_state_path)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )

    train_dataset = RLDSDataset(
        dataset_root,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
        image_aug=cfg.image_aug,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        normalization_stats=compatibility_statistics,
    )
    val_dataset = RLDSDataset(
        dataset_root,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=False,
        image_aug=False,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        normalization_stats=compatibility_statistics,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        validate_finite_values=cfg.validate_finite_values,
        expected_action_chunk_len=NUM_ACTIONS_CHUNK,
        expected_action_dim=ACTION_DIM,
        expected_proprio_dim=PROPRIO_DIM,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    runtime_document = build_runtime_resolved_config(
        cfg=cfg,
        vla=vla,
        action_head=action_head,
        proprio_projector=proprio_projector,
        model_path=model_path,
        split_path=split_path,
        split_document=split_document,
        stats_path=stats_path,
        stats_metadata=stats_metadata,
        stats_datasets=compatibility_statistics,
        device=device,
        validation_max_batches=validation_max_batches,
        validation_time_limit_seconds=validation_time_limit_seconds,
        checkpoint_steps=checkpoint_steps,
        initial_global_step=initial_global_step,
        launch_resolved_config_path=launch_resolved_config_path,
    )

    if is_main_process(distributed_state):
        write_json_atomic(runtime_config_path, runtime_document)
        write_json_atomic(
            run_dir / "dataset_statistics.json",
            compatibility_statistics,
        )
        write_json_atomic(
            run_dir / "dataset_statistics_provenance.json",
            provenance_statistics,
        )
        shutil.copy2(split_path, run_dir / "episode_split.json")

        if validation_history_path.exists():
            validation_history_path.unlink()
        for row in validation_history:
            append_jsonl(validation_history_path, row)

    distributed_barrier()

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    global_step = initial_global_step
    last_train_loss: Optional[float] = None
    latest_gradient_norm: Optional[float] = None
    training_started_at = time.time()
    status = "training_finished"
    failure_reason = ""
    caught_error: Optional[BaseException] = None

    vla.train()
    action_head.train()
    proprio_projector.train()
    optimizer.zero_grad(set_to_none=True)

    try:
        with tqdm.tqdm(
            total=cfg.max_steps,
            initial=initial_global_step,
            leave=False,
            disable=not is_main_process(distributed_state),
        ) as progress:
            for microbatch_index, batch in enumerate(train_dataloader):
                loss, metrics = official_l1_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    batch=batch,
                    device=device,
                    runtime_dtype=runtime_dtype,
                    num_patches=num_patches,
                    use_film=cfg.use_film,
                )
                (loss / cfg.grad_accumulation_steps).backward()

                for name, value in metrics.items():
                    if name in recent_metrics:
                        recent_metrics[name].append(value)

                if (microbatch_index + 1) % cfg.grad_accumulation_steps != 0:
                    continue

                next_global_step = global_step + 1
                log_due = (
                    is_main_process(distributed_state)
                    and next_global_step % cfg.console_log_interval_steps == 0
                )
                validation_due = should_run_validation(
                    next_global_step,
                    cfg.validation_start_step,
                    cfg.validation_interval_steps,
                )
                final_step_due = next_global_step >= cfg.max_steps

                if log_due or (
                    is_main_process(distributed_state)
                    and final_step_due
                ):
                    latest_gradient_norm = calculate_gradient_norm(
                        (vla, action_head, proprio_projector)
                    )

                if cfg.lr_warmup_steps > 0 and global_step < cfg.lr_warmup_steps:
                    warmup_progress = (global_step + 1) / cfg.lr_warmup_steps
                    warmup_lr = cfg.learning_rate * (
                        0.1 + 0.9 * warmup_progress
                    )
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = warmup_lr

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step = next_global_step
                progress.update(1)

                smoothed_metric_tensors = {
                    name: torch.stack(list(values)).mean()
                    for name, values in recent_metrics.items()
                    if values
                }
                should_sync_train_metrics = (
                    log_due
                    or validation_due
                    or final_step_due
                )
                smoothed_metrics = (
                    scalar_metrics_to_cpu(smoothed_metric_tensors)
                    if should_sync_train_metrics
                    else {}
                )
                if "loss_value" in smoothed_metrics:
                    last_train_loss = smoothed_metrics["loss_value"]
                current_lr = float(optimizer.param_groups[0]["lr"])

                if log_due:
                    elapsed_hours = (
                        time.time() - training_started_at
                    ) / 3600.0
                    train_row = {
                        "step": global_step,
                        "train_loss": last_train_loss,
                        "curr_action_l1_loss": smoothed_metrics.get(
                            "curr_action_l1_loss"
                        ),
                        "next_actions_l1_loss": smoothed_metrics.get(
                            "next_actions_l1_loss"
                        ),
                        "learning_rate": current_lr,
                        "gradient_norm": latest_gradient_norm,
                        "elapsed_time_hours": round(elapsed_hours, 6),
                        "peak_vram_allocated_gb": round(
                            torch.cuda.max_memory_allocated(device_id) / 1024**3,
                            6,
                        ),
                        "peak_vram_reserved_gb": round(
                            torch.cuda.max_memory_reserved(device_id) / 1024**3,
                            6,
                        ),
                    }
                    train_loss_curve.append(
                        {"step": global_step, "train_loss": last_train_loss}
                    )
                    learning_rate_curve.append(
                        {"step": global_step, "learning_rate": current_lr}
                    )
                    gradient_norm_curve.append(
                        {
                            "step": global_step,
                            "gradient_norm": latest_gradient_norm,
                        }
                    )
                    print(
                        "[DF-04-03 TRAIN] "
                        + " ".join(
                            f"{key}={value}" for key, value in train_row.items()
                        ),
                        flush=True,
                    )

                if validation_due:
                    validation_metrics = run_validation(
                        vla=vla,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        val_dataloader=val_dataloader,
                        device=device,
                        runtime_dtype=runtime_dtype,
                        num_patches=num_patches,
                        use_film=cfg.use_film,
                        max_batches=validation_max_batches,
                        time_limit_seconds=validation_time_limit_seconds,
                    )
                    validation_row: Dict[str, Any] = {
                        "created_at": utc_now(),
                        "step": global_step,
                        "validation_type": (
                            "full"
                            if validation_metrics["validation_completed"]
                            else "limited"
                        ),
                        "validation_completed": validation_metrics[
                            "validation_completed"
                        ],
                        "validation_stop_reason": validation_metrics[
                            "validation_stop_reason"
                        ],
                        "train_loss": last_train_loss,
                        "val_loss": validation_metrics["val_loss"],
                        "curr_action_l1_loss": validation_metrics.get(
                            "curr_action_l1_loss"
                        ),
                        "next_actions_l1_loss": validation_metrics.get(
                            "next_actions_l1_loss"
                        ),
                        "val_batches_count": validation_metrics[
                            "val_batches_count"
                        ],
                        "validation_time_seconds": validation_metrics[
                            "validation_time_seconds"
                        ],
                    }
                    validation_history.append(validation_row)
                    overfitting = analyse_overfitting(
                        validation_history,
                        cfg.overfit_patience,
                        cfg.overfit_min_delta,
                    )
                    validation_row["overfitting"] = overfitting
                    val_loss_curve.append(
                        {
                            "step": global_step,
                            "val_loss": validation_row["val_loss"],
                        }
                    )

                    improved = (
                        global_step >= cfg.best_model_start_step
                        and (
                            best_val_loss is None
                            or float(validation_row["val_loss"])
                            < float(best_val_loss) - cfg.overfit_min_delta
                        )
                    )
                    if improved:
                        best_val_loss = float(validation_row["val_loss"])
                        best_checkpoint_step = global_step

                    if is_main_process(distributed_state):
                        append_jsonl(validation_history_path, validation_row)
                        print(
                            "[DF-04-03 VAL] "
                            f"step={global_step} "
                            f"val_loss={validation_row['val_loss']} "
                            f"train_loss={last_train_loss} "
                            f"val_batches={validation_row['val_batches_count']} "
                            f"overfitting_detected="
                            f"{overfitting['overfitting_detected']} "
                            f"best_improved={improved}",
                            flush=True,
                        )
                        if improved:
                            best_metrics = dict(validation_row)
                            best_metrics["best_val_loss"] = best_val_loss
                            best_metrics[
                                "best_checkpoint_step"
                            ] = best_checkpoint_step
                            save_adapter_bundle(
                                output_dir=run_dir / "best_adapter",
                                vla=vla,
                                action_head=action_head,
                                proprio_projector=proprio_projector,
                                processor=processor,
                                compatibility_statistics=compatibility_statistics,
                                provenance_statistics=provenance_statistics,
                                metrics=best_metrics,
                            )
                    distributed_barrier()

                if global_step in checkpoint_steps:
                    checkpoint_dir = run_dir / f"checkpoint-{global_step}"
                    current_resume_checkpoints = list(resume_checkpoints)
                    if str(checkpoint_dir) not in current_resume_checkpoints:
                        current_resume_checkpoints.append(str(checkpoint_dir))

                    trainer_state = {
                        "best_val_loss": best_val_loss,
                        "best_checkpoint_step": best_checkpoint_step,
                        "train_loss_curve": train_loss_curve,
                        "val_loss_curve": val_loss_curve,
                        "learning_rate_curve": learning_rate_curve,
                        "gradient_norm_curve": gradient_norm_curve,
                        "validation_history": validation_history,
                        "resume_checkpoints": current_resume_checkpoints,
                    }
                    if is_main_process(distributed_state):
                        save_resume_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            global_step=global_step,
                            vla=vla,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            processor=processor,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            trainer_state=trainer_state,
                            compatibility_statistics=compatibility_statistics,
                            provenance_statistics=provenance_statistics,
                        )
                        resume_checkpoints = current_resume_checkpoints
                        prune_resume_checkpoints(run_dir, cfg.save_total_limit)
                        print(
                            "[DF-04-03 CHECKPOINT] "
                            f"step={global_step} path={checkpoint_dir}",
                            flush=True,
                        )
                    distributed_barrier()

                if global_step >= cfg.max_steps:
                    break

        if global_step < cfg.max_steps:
            raise RuntimeError(
                "Training dataloader ended before max_steps: "
                f"actual={global_step}, expected={cfg.max_steps}."
            )

        if is_main_process(distributed_state):
            save_adapter_bundle(
                output_dir=run_dir / "final_adapter",
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                processor=processor,
                compatibility_statistics=compatibility_statistics,
                provenance_statistics=provenance_statistics,
                metrics={
                    "global_step": global_step,
                    "final_train_loss": last_train_loss,
                    "best_val_loss": best_val_loss,
                    "best_checkpoint_step": best_checkpoint_step,
                },
            )
        distributed_barrier()

    except BaseException as error:
        status = "failed"
        failure_reason = f"{type(error).__name__}: {error}"
        caught_error = error

    training_time_hours = (time.time() - training_started_at) / 3600.0
    final_overfitting = analyse_overfitting(
        validation_history,
        cfg.overfit_patience,
        cfg.overfit_min_delta,
    )
    summary = {
        "created_at": utc_now(),
        "status": status,
        "failure_reason": failure_reason,
        "training_time_hours": round(training_time_hours, 6),
        "total_training_steps": global_step,
        "final_train_loss": last_train_loss,
        "best_val_loss": best_val_loss,
        "best_checkpoint_step": best_checkpoint_step,
        "gradient_norm": latest_gradient_norm,
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "learning_rate_curve": learning_rate_curve,
        "gradient_norm_curve": gradient_norm_curve,
        "validation_history": validation_history,
        "overfitting": final_overfitting,
        "best_adapter_path": str(run_dir / "best_adapter"),
        "final_adapter_path": str(run_dir / "final_adapter"),
        "resume_checkpoints": resume_checkpoints,
        "peak_vram_allocated_gb": round(
            torch.cuda.max_memory_allocated(device_id) / 1024**3,
            6,
        ),
        "peak_vram_reserved_gb": round(
            torch.cuda.max_memory_reserved(device_id) / 1024**3,
            6,
        ),
        "normalization_stats_json": str(stats_path),
        "normalization_stats_sha256": sha256_file(stats_path),
        "episode_split_json": str(split_path),
        "episode_split_sha256": sha256_file(split_path),
        "validation_start_step": cfg.validation_start_step,
        "validation_interval_steps": cfg.validation_interval_steps,
    }

    if is_main_process(distributed_state):
        write_json_atomic(training_summary_path, summary)
        print(
            "[DF-04-03 COMPLETE] "
            f"status={status} steps={global_step} "
            f"best_val_loss={best_val_loss} "
            f"best_checkpoint_step={best_checkpoint_step}",
            flush=True,
        )

    if caught_error is not None:
        raise caught_error


if __name__ == "__main__":
    finetune()
