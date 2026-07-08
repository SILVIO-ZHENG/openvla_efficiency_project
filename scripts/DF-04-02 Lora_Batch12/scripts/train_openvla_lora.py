"""
DF-04-02 LoRA Batch12 portable training wrapper.

This wrapper:
1. loads a portable YAML config;
2. resolves all relative paths against the uploaded experiment folder;
3. forces local/offline Hugging Face model loading;
4. passes LoRA, input, split, validation, checkpoint, proprioception,
   normalization, and logging settings to scripts/finetune.py;
5. reads the ACTUAL runtime configuration written by finetune.py;
6. writes reproducible metadata, Markdown, CSV, token-accuracy fields, and a launch-resolved config.

Important:
- Validation, best-adapter saving, resume checkpoints, action normalization,
  and the actual LoRAConfig construction are implemented in finetune.py.
- This wrapper verifies and records them; it does not silently copy YAML values
  into "actual runtime" metadata.
"""

import argparse
import copy
import csv
import json
import hashlib
import os
import platform
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


# ===========================================================================
# Global constants
# ===========================================================================

TASK_ID = "DF-04-02_LoRA"
DEFAULT_CONFIG_NAME = "train_config_lora_libero10_bs12_100k.yaml"

REQUIRED_BLOCKS = [
    "run",
    "paths",
    "model",
    "dataset",
    "normalization",
    "lora",
    "input",
    "training",
    "validation",
    "checkpointing",
    "logging",
]


# ===========================================================================
# Config loading and validation
# ===========================================================================

# Load the YAML file and reject malformed top-level structures early.
def load_config(config_path: Path) -> dict:
    """Load YAML and require a mapping at the root."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("The YAML root must be a mapping/dictionary.")
    return config


# Enforce required keys consistently and report the full dotted key name.
def require_keys(block_name: str, block: dict, keys: Iterable[str]) -> None:
    """Raise a clear error when required config keys are missing."""
    for key in keys:
        if key not in block:
            raise KeyError(f"Missing required config key: {block_name}.{key}")


# Validate both the YAML schema and the experiment-specific semantic rules.
def validate_config(config: dict) -> None:
    """Validate the portable LoRA YAML schema."""
    # Validate the outer schema first so later nested field access is safe.
    for block_name in REQUIRED_BLOCKS:
        if block_name not in config:
            raise KeyError(f"Missing required config block: {block_name}")
        if not isinstance(config[block_name], dict):
            raise TypeError(f"Config block must be a mapping: {block_name}")

    # Check required fields block by block to produce precise error messages.
    require_keys("run", config["run"], [
        "run_id",
        "branch_name",
        "random_seed",
    ])
    require_keys("paths", config["paths"], [
        "official_openvla_root",
        "official_finetune_py",
        "dataset_root",
        "split_json",
        "output_dir",
    ])
    require_keys("model", config["model"], [
        "hub_id",
        "local_path",
        "torch_dtype",
        "device",
        "local_files_only",
    ])
    require_keys("dataset", config["dataset"], [
        "mixture_name",
        "train_split_name",
        "val_split_name",
        "test_split_name",
        "expected_train_episodes",
        "expected_val_episodes",
        "expected_test_episodes",
        "split_seed",
    ])
    require_keys("normalization", config["normalization"], [
        "stats_json",
        "stats_source",
        "required_dataset_keys",
        "allow_pipeline_debug_stats",
    ])
    require_keys("lora", config["lora"], [
        "use_lora",
        "rank",
        "alpha",
        "dropout",
        "target_modules",
        "init_lora_weights",
    ])
    require_keys("input", config["input"], [
        "num_images_in_input",
        "use_wrist_image",
        "use_proprio",
    ])
    require_keys("training", config["training"], [
        "objective",
        "use_l1_regression",
        "use_action_head",
        "batch_size",
        "max_steps",
        "learning_rate",
        "lr_warmup_steps",
        "num_steps_before_decay",
        "grad_accumulation_steps",
        "image_aug",
        "shuffle_buffer_size",
        "use_quantization",
        "validate_finite_values",
    ])
    require_keys("validation", config["validation"], [
        "start_step",
        "interval_steps",
        "max_batches",
        "time_limit_seconds",
        "best_model_start_step",
        "overfit_patience",
        "overfit_min_delta",
    ])
    require_keys("checkpointing", config["checkpointing"], [
        "resume_checkpoint_steps",
        "save_total_limit",
        "resume_from_checkpoint",
    ])
    require_keys("logging", config["logging"], [
        "console_log_interval_steps",
        "wandb_project",
    ])

    # Semantic checks.
    if not isinstance(config["lora"]["use_lora"], bool):
        raise TypeError("lora.use_lora must be a boolean.")
    if not config["lora"]["use_lora"]:
        raise ValueError("DF-04-02 requires lora.use_lora=true.")

    if not bool(config["model"]["local_files_only"]):
        raise ValueError(
            "model.local_files_only must be true. "
            "This experiment must load the already-downloaded local model."
        )

    if str(config["dataset"]["train_split_name"]) != "train":
        raise ValueError("dataset.train_split_name must be 'train'.")

    if int(config["validation"]["start_step"]) < 0:
        raise ValueError("validation.start_step cannot be negative.")
    if int(config["validation"]["interval_steps"]) <= 0:
        raise ValueError("validation.interval_steps must be greater than 0.")
    if int(config["validation"]["best_model_start_step"]) < 0:
        raise ValueError("validation.best_model_start_step cannot be negative.")

    if str(config["training"]["objective"]) != "token_cross_entropy":
        raise ValueError("training.objective must be 'token_cross_entropy'.")
    if bool(config["training"]["use_l1_regression"]):
        raise ValueError("training.use_l1_regression must be false for this LoRA run.")
    if bool(config["training"]["use_action_head"]):
        raise ValueError("training.use_action_head must be false for this LoRA run.")
    if not bool(config["input"]["use_proprio"]):
        raise ValueError("input.use_proprio must be true for this run.")
    if bool(config["input"]["use_wrist_image"]) and int(config["input"]["num_images_in_input"]) < 2:
        raise ValueError(
            "input.num_images_in_input must be at least 2 when input.use_wrist_image is true."
        )

    if not str(config["run"]["run_id"]).strip():
        raise ValueError("run.run_id cannot be empty.")
    if not str(config["run"]["branch_name"]).strip():
        raise ValueError("run.branch_name cannot be empty.")
    try:
        int(config["run"]["random_seed"])
    except (TypeError, ValueError) as error:
        raise ValueError("run.random_seed must be an integer.") from error


# ===========================================================================
# Path resolution
# ===========================================================================

# Expand user and environment syntax, then anchor relative paths to the bundle.
def expand_path(value: str, bundle_root: Path) -> Path:
    """Expand ~ and environment variables, then resolve relative paths."""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = bundle_root / path
    return path.resolve()


# Create a portable launch configuration with absolute paths and derived values.
def resolve_config(raw_config: dict, config_path: Path) -> dict:
    """
    Produce the launch-resolved config.

    Resolves paths and computes derived launch values.
    The actual runtime model/LoRA config is written by finetune.py after
    objects are constructed; this function does not claim to know those values.
    """
    # Preserve the raw YAML object and modify only the resolved copy.
    config = copy.deepcopy(raw_config)
    bundle_root = Path(__file__).resolve().parent.parent

    paths = config["paths"]
    paths["project_root"] = str(bundle_root)
    paths["config_path"] = str(config_path.resolve())

    # Resolve source, dataset, split, and output paths before launching training.
    for key in [
        "official_openvla_root",
        "official_finetune_py",
        "dataset_root",
        "split_json",
        "output_dir",
    ]:
        paths[key] = str(expand_path(paths[key], bundle_root))

    config["model"]["local_path"] = str(
        expand_path(config["model"]["local_path"], bundle_root)
    )
    config["normalization"]["stats_json"] = str(
        expand_path(config["normalization"]["stats_json"], bundle_root)
    )

    output_dir = Path(paths["output_dir"])
    paths.setdefault("log_path", str(output_dir / "df_04_02_lora_training.md"))
    paths.setdefault("summary_csv", str(output_dir / "df_04_02_lora_summary.csv"))
    paths.setdefault("metadata_json", str(output_dir / "df_04_02_lora_metadata.json"))
    paths.setdefault("launch_resolved_config_yaml", str(output_dir / "launch_resolved_config.yaml"))
    paths.setdefault("runtime_resolved_config_json", str(output_dir / "runtime_resolved_config.json"))
    paths.setdefault("training_summary_json", str(output_dir / "training_summary.json"))
    paths.setdefault("validation_history_jsonl", str(output_dir / "validation_history.jsonl"))

    for key in [
        "log_path",
        "summary_csv",
        "metadata_json",
        "launch_resolved_config_yaml",
        "runtime_resolved_config_json",
        "training_summary_json",
        "validation_history_jsonl",
    ]:
        paths[key] = str(expand_path(paths[key], bundle_root))

    # Effective batch size includes per-device batch, accumulation, and process count.
    world_size = int(config["training"].get("world_size", 1))
    config["training"]["world_size"] = world_size
    config["training"]["effective_global_batch_size"] = (
        int(config["training"]["batch_size"])
        * int(config["training"]["grad_accumulation_steps"])
        * world_size
    )

    config["derived"] = {
        "task_id": TASK_ID,
        "bundle_root": str(bundle_root),
        "model_source": "local_huggingface_snapshot",
        "offline_mode": True,
        "best_adapter_dir": str(output_dir / "best_adapter"),
        "final_adapter_dir": str(output_dir / "final_adapter"),
        "checkpoint_pattern": str(output_dir / "checkpoint-{step}"),
    }

    return config


# ===========================================================================
# Environment setup
# ===========================================================================

# Seed every available random-number generator used by the wrapper or trainer.
def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    # NumPy and PyTorch remain optional so lightweight preflight checks can still run.
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# Check whether required Python packages can be imported without loading training data.
def check_dependencies() -> dict:
    """Record dependency availability."""
    status: Dict[str, str] = {}
    for package_name in [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "yaml",
        "tensorflow",
        "tensorflow_datasets",
    ]:
        try:
            __import__(package_name)
            status[package_name] = "installed"
        except ImportError:
            status[package_name] = "not_installed"
    return status


# Collect basic CUDA and GPU metadata visible to the wrapper process.
def get_gpu_info() -> dict:
    """
    Record basic GPU information visible from the wrapper process.

    Training peak VRAM is written by finetune.py because training runs in
    a subprocess; this function does not claim the training peak.
    """
    info = {
        "cuda_available": False,
        "gpu_name": "not_available",
        "gpu_count": 0,
        "gpu_total_vram_gb": None,
        "torch_version": "not_available",
        "cuda_version": "not_available",
    }
    # Import PyTorch lazily because metadata collection should not force installation.
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = str(torch.version.cuda)
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 4
            )
    except ImportError:
        pass
    return info


# Create the run directory and every parent directory needed by output files.
def prepare_output_dirs(config: dict) -> None:
    """Create all lightweight output parents and the run directory."""
    paths = config["paths"]
    Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)
    for key in [
        "log_path",
        "summary_csv",
        "metadata_json",
        "launch_resolved_config_yaml",
        "runtime_resolved_config_json",
        "training_summary_json",
        "validation_history_jsonl",
    ]:
        Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Preflight utilities
# ===========================================================================

# Hash a file in chunks so large files do not need to be loaded fully into memory.
def sha256_file(path: Path) -> str:
    """Compute a reproducibility hash without loading the whole file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Recursively collect dictionary keys from nested dictionaries and lists.
def collect_json_keys(value: Any) -> set:
    """Collect all dictionary keys recursively for flexible stats validation."""
    keys: set = set()
    if isinstance(value, dict):
        for key, nested_value in value.items():
            keys.add(str(key))
            keys.update(collect_json_keys(nested_value))
    elif isinstance(value, list):
        for item in value:
            keys.update(collect_json_keys(item))
    return keys


# Verify that the configured local Hugging Face snapshot is complete enough to load.
def check_local_model(model_path: Path) -> dict:
    """Require an already-downloaded Hugging Face snapshot."""
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    config_json = model_path / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"Local model config.json was not found: {config_json}")
    # Accept single-file weights, sharded weights, and their index files.
    weight_files = (
        list(model_path.glob("*.safetensors"))
        + list(model_path.glob("*.bin"))
        + list(model_path.glob("*.index.json"))
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No Hugging Face model weight/index files were found in: {model_path}"
        )
    return {
        "local_model_path": str(model_path),
        "config_json": str(config_json),
        "weight_file_count": len(weight_files),
    }


# Load the fixed episode split and verify its totals against the YAML contract.
def load_and_validate_split(config: dict) -> dict:
    """Validate the fixed logical split against expected episode counts."""
    split_path = Path(config["paths"]["split_json"])
    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON does not exist: {split_path}")
    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)
    # The split file is the source of truth; YAML values are expected contracts.
    totals = split_data.get("totals", {})
    expected = {
        "train_count": int(config["dataset"]["expected_train_episodes"]),
        "val_count": int(config["dataset"]["expected_val_episodes"]),
        "test_count": int(config["dataset"]["expected_test_episodes"]),
    }
    for key, expected_value in expected.items():
        actual_value = totals.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Split mismatch for {key}: actual={actual_value}, "
                f"expected={expected_value}"
            )
    return split_data


# Validate provenance, dataset coverage, and tensor dimensions in the statistics file.
def validate_normalization_stats(config: dict) -> dict:
    """
    Validate formal LIBERO normalization statistics.

    The statistics file must:
    1. contain explicit provenance metadata;
    2. contain all required LIBERO dataset keys;
    3. not be marked as pipeline-debug-only for formal training;
    4. not contain LIBERO values copied directly from bridge_orig.
    """
    normalization_config = config["normalization"]
    stats_path = Path(normalization_config["stats_json"])

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization statistics JSON does not exist: {stats_path}"
        )

    with stats_path.open("r", encoding="utf-8") as file:
        stats_data = json.load(file)

    if not isinstance(stats_data, dict):
        raise TypeError("Normalization statistics JSON root must be a dictionary.")

    # Keep provenance metadata separate from per-dataset numerical statistics.
    metadata = stats_data.get("metadata")
    datasets = stats_data.get("datasets")

    if not isinstance(metadata, dict):
        raise KeyError("Statistics JSON is missing the required 'metadata' block.")
    if not isinstance(datasets, dict):
        raise KeyError("Statistics JSON is missing the required 'datasets' block.")

    required_dataset_keys = [
        str(key) for key in normalization_config["required_dataset_keys"]
    ]
    missing_dataset_keys = [k for k in required_dataset_keys if k not in datasets]
    if missing_dataset_keys:
        raise KeyError(
            "Normalization statistics are missing dataset keys: "
            + ", ".join(missing_dataset_keys)
        )

    if "pipeline_debug_only" not in metadata:
        raise KeyError("Statistics metadata is missing 'pipeline_debug_only'.")

    pipeline_debug_only = metadata["pipeline_debug_only"]
    if not isinstance(pipeline_debug_only, bool):
        raise TypeError("metadata.pipeline_debug_only must be true or false.")

    allow_debug_stats = bool(normalization_config["allow_pipeline_debug_stats"])
    if pipeline_debug_only and not allow_debug_stats:
        raise ValueError(
            "The statistics file is marked pipeline_debug_only=true "
            "and cannot be used for formal training."
        )

    stats_source = str(normalization_config["stats_source"])
    actual_stats_source = metadata.get("stats_source")
    if actual_stats_source != stats_source:
        raise ValueError(
            "Normalization statistics source mismatch: "
            f"actual={actual_stats_source}, expected={stats_source}"
        )

    # Verify LIBERO action statistics are not copied directly from bridge_orig.
    if "bridge_orig" in datasets and not allow_debug_stats:
        bridge_action = datasets["bridge_orig"].get("action")
        duplicated_keys = [
            dataset_key
            for dataset_key in required_dataset_keys
            if (
                bridge_action is not None
                and datasets[dataset_key].get("action") is not None
                and datasets[dataset_key].get("action") == bridge_action
            )
        ]
        if duplicated_keys:
            raise ValueError(
                "The following LIBERO action statistics are identical "
                "to bridge_orig and may be placeholders: "
                + ", ".join(duplicated_keys)
            )

    # Validate action statistics block dimensions.
    # Every LIBERO action-statistics field must describe exactly seven dimensions.
    required_action_fields = ["mean", "std", "min", "max", "q01", "q99", "mask"]
    for dataset_key in required_dataset_keys:
        dataset_stats = datasets[dataset_key]
        if not isinstance(dataset_stats, dict):
            raise TypeError(f"Statistics for {dataset_key} must be a dictionary.")
        action_stats = dataset_stats.get("action")
        if not isinstance(action_stats, dict):
            raise KeyError(f"{dataset_key} is missing the 'action' statistics block.")
        missing_action_fields = [
            f for f in required_action_fields if f not in action_stats
        ]
        if missing_action_fields:
            raise KeyError(
                f"{dataset_key}.action is missing fields: "
                + ", ".join(missing_action_fields)
            )
        for field in required_action_fields:
            values = action_stats[field]
            if not isinstance(values, list) or len(values) != 7:
                raise ValueError(
                    f"{dataset_key}.action.{field} must contain 7 values."
                )

    return {
        "stats_json": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "stats_source": actual_stats_source,
        "pipeline_debug_only": pipeline_debug_only,
        "required_dataset_keys": required_dataset_keys,
    }


# Confirm that the patched OpenVLA source exposes every feature required by this run.
def check_source_contract(config: dict) -> dict:
    """
    Check that the bundled source exposes the features this wrapper passes.

    This is a static safety check. The runtime JSON written by finetune.py
    later proves which values were actually used to construct LoRAConfig and
    the trainer.
    """
    source_root = Path(config["paths"]["official_openvla_root"])
    finetune_path = Path(config["paths"]["official_finetune_py"])
    batch_dataset_path = source_root / "prismatic/vla/datasets/datasets.py"
    rlds_dataset_path = source_root / "prismatic/vla/datasets/rlds/dataset.py"
    mixtures_path = source_root / "prismatic/vla/datasets/rlds/oxe/mixtures.py"

    for path in [finetune_path, batch_dataset_path, rlds_dataset_path, mixtures_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required source file was not found: {path}")

    # Static token checks catch missing source patches before expensive training starts.
    finetune_text = finetune_path.read_text(encoding="utf-8", errors="ignore")
    batch_dataset_text = batch_dataset_path.read_text(encoding="utf-8", errors="ignore")
    rlds_dataset_text = rlds_dataset_path.read_text(encoding="utf-8", errors="ignore")
    mixtures_text = mixtures_path.read_text(encoding="utf-8", errors="ignore")

    # These tokens must exist in the patched finetune.py.
    required_finetune_tokens = [
        "lora_alpha",
        "lora_target_modules",
        "runtime_resolved_config",
        "validation_start_step",
        "validation_interval_steps",
        "best_model_start_step",
        "resume_checkpoint_steps",
        "normalization_stats_json",
        "training_objective",
        "use_l1_regression",
        "use_action_head",
        "token_cross_entropy_forward_pass",
        "compute_action_token_accuracy",
        "action_token_accuracy",
        "ONE_STEP_ACTION_CHUNK_LEN",
        "use_proprio",
    ]
    missing_finetune_tokens = [
        token for token in required_finetune_tokens if token not in finetune_text
    ]
    if missing_finetune_tokens:
        raise RuntimeError(
            "scripts/finetune.py has not yet been patched for: "
            + ", ".join(missing_finetune_tokens)
        )

    forbidden_finetune_tokens = [
        "official_l1_forward_pass",
        "L1RegressionActionHead",
        "action_head.pt",
    ]
    present_forbidden_tokens = [
        token for token in forbidden_finetune_tokens if token in finetune_text
    ]
    if present_forbidden_tokens:
        raise RuntimeError(
            "scripts/finetune.py still contains OFT action-head tokens: "
            + ", ".join(present_forbidden_tokens)
        )

    required_batch_dataset_tokens = [
        "One-step action-token supervision",
        "Expected one-step action shape",
        '"future_action_window_size": 0',
    ]
    missing_batch_dataset_tokens = [
        token for token in required_batch_dataset_tokens
        if token not in batch_dataset_text
    ]
    if missing_batch_dataset_tokens:
        raise RuntimeError(
            "src/prismatic/vla/datasets/datasets.py is not patched for one-step CE: "
            + ", ".join(missing_batch_dataset_tokens)
        )

    if "NUM_ACTIONS_CHUNK" in batch_dataset_text:
        raise RuntimeError(
            "src/prismatic/vla/datasets/datasets.py still imports or uses NUM_ACTIONS_CHUNK."
        )

    if "OPENVLA_EPISODE_SPLIT_JSON" not in rlds_dataset_text:
        raise RuntimeError(
            "The fixed logical split patch OPENVLA_EPISODE_SPLIT_JSON "
            "was not found in rlds/dataset.py."
        )

    if "train[:95%]" in rlds_dataset_text or "train[:95%]" in finetune_text:
        raise RuntimeError(
            "A forbidden temporary split 'train[:95%]' was found in source."
        )

    mixture_name = str(config["dataset"]["mixture_name"])
    if mixture_name not in mixtures_text:
        raise RuntimeError(f"Dataset mixture was not found: {mixture_name}")

    return {
        "finetune_path": str(finetune_path),
        "batch_dataset_path": str(batch_dataset_path),
        "rlds_dataset_path": str(rlds_dataset_path),
        "mixtures_path": str(mixtures_path),
        "source_contract_passed": True,
    }


# Run all inexpensive safety checks before allocating time on the training GPU.
def run_preflight_checks(config: dict) -> dict:
    """Run all checks required before launching expensive GPU training."""
    model_info = check_local_model(Path(config["model"]["local_path"]))
    split_data = load_and_validate_split(config)
    normalization_info = validate_normalization_stats(config)
    source_info = check_source_contract(config)
    return {
        "model": model_info,
        "split_data": split_data,
        "normalization": normalization_info,
        "source": source_info,
    }


# ===========================================================================
# Command building
# ===========================================================================

# Convert target-module settings into the single string expected by the CLI.
def serialise_target_modules(value: Any) -> str:
    """Convert YAML string/list target modules to one CLI string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ",".join(value)
    raise TypeError("lora.target_modules must be a string or a list of strings.")


# Convert truthy values into the exact capitalized boolean spelling used by the parser.
def bool_text(value: Any) -> str:
    """Convert a Python boolean to the capitalized string the CLI parser expects."""
    return "True" if bool(value) else "False"


# Convert optional values into the explicit sentinel accepted by the patched trainer.
def optional_cli_value(value: Any) -> str:
    """Represent None as the explicit string 'none' for the patched trainer."""
    return "none" if value is None else str(value)


# Translate the resolved configuration into the complete torchrun argument vector.
def build_finetune_command(config: dict) -> List[str]:
    """Build the complete torchrun command for the patched finetune.py."""
    run = config["run"]
    paths = config["paths"]
    model = config["model"]
    dataset = config["dataset"]
    normalization = config["normalization"]
    lora = config["lora"]
    input_config = config["input"]
    training = config["training"]
    validation = config["validation"]
    checkpointing = config["checkpointing"]
    logging_config = config["logging"]

    world_size = int(training["world_size"])

    # Optional limits use the trainer's explicit 'none' sentinel rather than omission.
    validation_max_batches = (
        "none" if validation["max_batches"] is None
        else str(validation["max_batches"])
    )
    validation_time_limit_seconds = (
        "none" if validation["time_limit_seconds"] is None
        else str(validation["time_limit_seconds"])
    )
    resume_from_checkpoint = (
        "none" if checkpointing["resume_from_checkpoint"] is None
        else str(checkpointing["resume_from_checkpoint"])
    )

    return [
        "torchrun",
        "--standalone",
        "--nnodes", "1",
        "--nproc-per-node", str(world_size),

        paths["official_finetune_py"],

        # Local model configuration.
        "--vla_path", model["local_path"],
        "--model_hub_id", model["hub_id"],
        "--local_files_only", bool_text(model["local_files_only"]),
        "--torch_dtype", model["torch_dtype"],
        "--device", model["device"],

        # Dataset and fixed logical split.
        "--data_root_dir", paths["dataset_root"],
        "--dataset_name", dataset["mixture_name"],
        "--episode_split_json", paths["split_json"],
        "--train_split_name", dataset["train_split_name"],
        "--val_split_name", dataset["val_split_name"],
        "--test_split_name", dataset["test_split_name"],
        "--expected_train_episodes", str(dataset["expected_train_episodes"]),
        "--expected_val_episodes", str(dataset["expected_val_episodes"]),
        "--expected_test_episodes", str(dataset["expected_test_episodes"]),
        "--split_seed", str(dataset["split_seed"]),

        # External train-only normalization statistics.
        "--normalization_stats_json", normalization["stats_json"],
        "--normalization_stats_source", normalization["stats_source"],
        "--allow_pipeline_debug_stats",
        bool_text(normalization["allow_pipeline_debug_stats"]),

        # Output paths.
        "--run_root_dir", paths["output_dir"],
        "--runtime_resolved_config_path", paths["runtime_resolved_config_json"],
        "--training_summary_path", paths["training_summary_json"],
        "--validation_history_path", paths["validation_history_jsonl"],
        "--launch_resolved_config_path", paths["launch_resolved_config_yaml"],

        # Input and action-target configuration.
        "--training_objective", str(training["objective"]),
        "--use_l1_regression", bool_text(training["use_l1_regression"]),
        "--use_action_head", bool_text(training["use_action_head"]),
        "--use_diffusion", "False",
        "--use_film", "False",
        "--num_images_in_input", str(input_config["num_images_in_input"]),
        "--use_proprio", bool_text(input_config["use_proprio"]),

        # Optimization configuration.
        "--batch_size", str(training["batch_size"]),
        "--max_steps", str(training["max_steps"]),
        "--learning_rate", str(training["learning_rate"]),
        "--lr_warmup_steps", str(training["lr_warmup_steps"]),
        "--num_steps_before_decay", str(training["num_steps_before_decay"]),
        "--grad_accumulation_steps", str(training["grad_accumulation_steps"]),
        "--image_aug", bool_text(training["image_aug"]),
        "--shuffle_buffer_size", str(training["shuffle_buffer_size"]),
        "--use_quantization", bool_text(training["use_quantization"]),
        "--validate_finite_values", bool_text(training["validate_finite_values"]),
        "--random_seed", str(run["random_seed"]),

        # Runtime LoRA configuration.
        "--use_lora", bool_text(lora["use_lora"]),
        "--lora_rank", str(lora["rank"]),
        "--lora_alpha", str(lora["alpha"]),
        "--lora_dropout", str(lora["dropout"]),
        # serialise_target_modules handles both string and list correctly.
        "--lora_target_modules", serialise_target_modules(lora["target_modules"]),
        "--init_lora_weights", str(lora["init_lora_weights"]),

        # Training-console logging.
        "--console_log_interval_steps",
        str(logging_config["console_log_interval_steps"]),

        # Scheduled validation.
        "--validation_start_step", str(validation["start_step"]),
        "--validation_interval_steps", str(validation["interval_steps"]),
        "--validation_max_batches", validation_max_batches,
        "--validation_time_limit_seconds", validation_time_limit_seconds,
        "--best_model_start_step", str(validation["best_model_start_step"]),
        "--overfit_patience", str(validation["overfit_patience"]),
        "--overfit_min_delta", str(validation["overfit_min_delta"]),

        # Resumable checkpoint configuration.
        "--resume_checkpoint_steps", str(checkpointing["resume_checkpoint_steps"]),
        "--save_total_limit", str(checkpointing["save_total_limit"]),
        "--resume_from_checkpoint", resume_from_checkpoint,

        # Metadata-only values.
        "--wandb_project", str(logging_config["wandb_project"]),
        "--run_id_note", str(run["run_id"]),
    ]


# Build an isolated subprocess environment with offline and fixed-split settings.
def build_environment(config: dict) -> dict:
    """Build an offline, split-aware environment for the trainer subprocess."""
    env = os.environ.copy()
    source_root = Path(config["paths"]["official_openvla_root"]).resolve()
    existing_pythonpath = env.get("PYTHONPATH", "")

    # Put the selected OpenVLA tree first so the subprocess imports that source copy.
    env["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else f"{source_root}{os.pathsep}{existing_pythonpath}"
    )

    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["OPENVLA_EPISODE_SPLIT_JSON"] = config["paths"]["split_json"]
    env["OPENVLA_EPISODE_SPLIT_NAME"] = config["dataset"]["train_split_name"]
    env["OPENVLA_CONSOLE_LOG_INTERVAL_STEPS"] = str(
        config["logging"]["console_log_interval_steps"]
    )
    return env


# ===========================================================================
# Git
# ===========================================================================

# Read the current Git revision for reproducibility without modifying repository state.
def get_git_commit_hash(project_root: Path) -> Optional[str]:
    """Read the current commit hash without changing Git state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


# ===========================================================================
# Launch config persistence
# ===========================================================================

# Persist the exact launch-time configuration separately from runtime-discovered values.
def save_launch_resolved_config(
    config: dict,
    command: List[str],
    preflight: dict,
    git_commit_hash: Optional[str],
) -> None:
    """
    Save the genuinely resolved launch configuration.

    Actual model/LoRA/trainer values are separately written by finetune.py to
    runtime_resolved_config.json after runtime object construction.
    """
    # Add launch evidence without mutating the in-memory resolved configuration.
    document = copy.deepcopy(config)
    totals = preflight["split_data"].get("totals", {})

    document["resolved_launch"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command_argv": command,
        "command_shell": shlex.join(command),
        "git_commit_hash": git_commit_hash,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "split_totals": totals,
        "normalization_stats_sha256": preflight["normalization"]["stats_sha256"],
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "OPENVLA_EPISODE_SPLIT_JSON": config["paths"]["split_json"],
            "OPENVLA_EPISODE_SPLIT_NAME": config["dataset"]["train_split_name"],
        },
    }

    output_path = Path(config["paths"]["launch_resolved_config_yaml"])
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, sort_keys=False, allow_unicode=True)


# ===========================================================================
# JSON/JSONL loaders
# ===========================================================================

# Load a JSON object and optionally require it as evidence of a successful run.
def load_json(path: Path, required: bool = False) -> dict:
    """Load a JSON object from disk, optionally requiring that it exists."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required runtime JSON was not produced: {path}")
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in: {path}")
    return data


# Parse newline-delimited validation records while reporting the exact failing line.
def load_jsonl(path: Path) -> List[dict]:
    """Load validation history from a JSONL file, one JSON object per line."""
    if not path.exists():
        return []
    rows: List[dict] = []
    # Parse each record independently so a corrupt line is reported precisely.
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if isinstance(item, dict):
                rows.append(item)
    return rows


# ===========================================================================
# Overfitting analysis
# ===========================================================================

# Derive a conservative overfitting signal from recent training and validation losses.
def analyse_overfitting(history: List[dict], config: dict) -> dict:
    """
    Produce a simple overfitting signal from validation history.

    Overfitting is flagged when validation loss rises, training loss falls,
    and recent validation points fail to beat the previous best.
    """
    validation_config = config["validation"]
    start_step = int(validation_config["best_model_start_step"])
    patience = int(validation_config["overfit_patience"])
    min_delta = float(validation_config["overfit_min_delta"])

    # Ignore validation points outside the best-model decision window.
    eligible = [
        row for row in history
        if int(row.get("step", -1)) >= start_step
        and row.get("val_loss") is not None
    ]

    # Prefer full validation runs when enough points exist for a stable decision.
    full_points = [row for row in eligible if row.get("validation_type") == "full"]
    points = full_points if len(full_points) >= patience else eligible

    result = {
        "overfitting_checked": len(points) >= patience,
        "overfitting_detected": False,
        "overfitting_reason": "insufficient_validation_points",
        "overfitting_points_used": len(points),
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

    recent_val = [float(row["val_loss"]) for row in recent]
    recent_train = [
        float(row["train_loss"]) for row in recent
        if row.get("train_loss") is not None
    ]

    no_recent_improvement = min(recent_val) >= previous_best - min_delta
    val_trending_up = recent_val[-1] > recent_val[0] + min_delta
    train_trending_down = (
        len(recent_train) >= 2 and recent_train[-1] < recent_train[0]
    )

    # Require all three signals to reduce false positives from noisy losses.
    result["overfitting_detected"] = bool(
        no_recent_improvement and val_trending_up and train_trending_down
    )
    result["overfitting_reason"] = (
        "validation_loss_rising_while_train_loss_falling"
        if result["overfitting_detected"]
        else "no_clear_overfitting_signal"
    )
    result["previous_best_val_loss"] = previous_best
    result["recent_val_losses"] = recent_val
    result["recent_train_losses"] = recent_train
    return result


# ===========================================================================
# Training subprocess
# ===========================================================================

# Launch training, stream logs, and collect all structured outputs produced afterward.
def run_lora_training(config: dict) -> Dict[str, Any]:
    """Launch the patched trainer and then read structured runtime outputs."""
    command = build_finetune_command(config)
    env = build_environment(config)
    stdout_log_path = Path(config["paths"]["output_dir"]) / "training.stdout.log"

    start_time = time.time()

    # Mirror subprocess output to both the terminal and a persistent log file.
    with stdout_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(shlex.join(command) + "\n\n")
        log_file.write("Offline model loading: enabled\n")
        log_file.write(f"Local model path: {config['model']['local_path']}\n")
        log_file.write(f"Split JSON: {env['OPENVLA_EPISODE_SPLIT_JSON']}\n")
        log_file.write(f"Train split name: {env['OPENVLA_EPISODE_SPLIT_NAME']}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            # cwd must be the OpenVLA source root so finetune.py's relative
            # imports and any relative resource paths work correctly.
            cwd=config["paths"]["official_openvla_root"],
            text=True,
            bufsize=1,
        )

        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

        return_code = process.wait()

    training_time_hours = round((time.time() - start_time) / 3600, 4)

    # A successful run must produce runtime evidence and a structured summary.
    runtime_config = load_json(
        Path(config["paths"]["runtime_resolved_config_json"]),
        required=(return_code == 0),
    )
    training_summary = load_json(
        Path(config["paths"]["training_summary_json"]),
        required=(return_code == 0),
    )
    validation_history = load_jsonl(
        Path(config["paths"]["validation_history_jsonl"])
    )
    overfitting = analyse_overfitting(validation_history, config)

    return {
        "returncode": return_code,
        "training_time_hours": training_time_hours,
        "training_stdout_log": str(stdout_log_path),
        "command_argv": command,
        "command": shlex.join(command),
        "runtime_config": runtime_config,
        "training_summary": training_summary,
        "validation_history": validation_history,
        "overfitting": overfitting,
    }


# ===========================================================================
# Metadata assembly
# ===========================================================================

# Extract the actual LoRA configuration recorded after runtime object construction.
def runtime_lora_config(training_result: dict) -> dict:
    """Read actual LoRAConfig values written by finetune.py."""
    runtime = training_result.get("runtime_config", {})
    value = runtime.get("lora_config", {})
    return value if isinstance(value, dict) else {}


# Assemble one reproducibility record from requested settings and runtime truth.
def build_metadata(
    config: dict,
    dependency_status: dict,
    gpu_info: dict,
    preflight: Optional[dict],
    status: str,
    training_result: Dict[str, Any],
    failure_reason: str,
    git_commit_hash: Optional[str],
) -> dict:
    """Build metadata using runtime truth, not copied YAML claims."""
    split_totals = (
        preflight["split_data"].get("totals", {}) if preflight else {}
    )
    # Runtime-resolved values take precedence over YAML requests for actual metadata.
    runtime = training_result.get("runtime_config", {})
    summary = training_result.get("training_summary", {})
    actual_lora = runtime_lora_config(training_result)
    actual_model = runtime.get("model", {}) if isinstance(runtime, dict) else {}
    actual_dataset = runtime.get("dataset", {}) if isinstance(runtime, dict) else {}
    actual_input = runtime.get("input", {}) if isinstance(runtime, dict) else {}
    actual_objective = (
        runtime.get("training_objective", {}) if isinstance(runtime, dict) else {}
    )

    # trainable_params lives under runtime["parameters"]["trainable_params"]
    # in the new finetune.py schema.
    actual_parameters = (
        runtime.get("parameters", {}) if isinstance(runtime, dict) else {}
    )

    # Keep requested settings visible while recording actual values from runtime files.
    return {
        "task_id": TASK_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": config["run"]["run_id"],
        "branch_name": config["run"]["branch_name"],
        "git_commit_hash": git_commit_hash,
        "random_seed": config["run"]["random_seed"],
        "status": status,
        "failure_reason": failure_reason,
        "returncode": training_result.get("returncode"),
        "model_hub_id": config["model"]["hub_id"],
        "requested_model_local_path": config["model"]["local_path"],
        "actual_model_loaded_from": actual_model.get("loaded_from"),
        "actual_local_files_only": actual_model.get("local_files_only"),
        "torch_dtype": config["model"]["torch_dtype"],
        "device": config["model"]["device"],
        "dataset_name": config["dataset"]["mixture_name"],
        "split_json": config["paths"]["split_json"],
        "train_count": split_totals.get("train_count"),
        "val_count": split_totals.get("val_count"),
        "test_count": split_totals.get("test_count"),
        "actual_dataset_runtime": actual_dataset,
        "normalization_stats_json": config["normalization"]["stats_json"],
        "normalization_stats_source_requested": config["normalization"]["stats_source"],
        "normalization_stats_source_actual": actual_dataset.get(
            "normalization_stats_source"
        ),
        "training_time_hours": training_result.get("training_time_hours"),
        "total_training_steps": summary.get("total_training_steps"),
        "final_train_loss": summary.get("final_train_loss"),
        "final_train_action_token_accuracy": summary.get(
            "final_train_action_token_accuracy"
        ),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_action_token_accuracy": summary.get(
            "best_val_action_token_accuracy"
        ),
        "best_checkpoint_step": summary.get("best_checkpoint_step"),
        "gradient_norm": summary.get("gradient_norm"),
        "train_loss_curve": summary.get("train_loss_curve", []),
        "val_loss_curve": summary.get("val_loss_curve", []),
        "train_action_token_accuracy_curve": summary.get(
            "train_action_token_accuracy_curve", []
        ),
        "val_action_token_accuracy_curve": summary.get(
            "val_action_token_accuracy_curve", []
        ),
        "learning_rate_curve": summary.get("learning_rate_curve", []),
        "learning_rate": config["training"]["learning_rate"],
        "per_device_batch_size": config["training"]["batch_size"],
        "effective_global_batch_size": config["training"]["effective_global_batch_size"],
        "max_steps": config["training"]["max_steps"],
        "grad_accumulation_steps": config["training"]["grad_accumulation_steps"],
        "training_objective": config["training"]["objective"],
        "use_l1_regression": config["training"]["use_l1_regression"],
        "use_action_head": config["training"]["use_action_head"],
        "num_images_in_input": config["input"]["num_images_in_input"],
        "use_wrist_image": config["input"]["use_wrist_image"],
        "use_proprio": config["input"]["use_proprio"],
        # Read trainable_params from the nested parameters block.
        "trainable_params": actual_parameters.get("trainable_params"),
        "total_params": actual_parameters.get("total_params"),
        "requested_lora_config": config["lora"],
        "actual_lora_rank": actual_lora.get("r"),
        "actual_lora_alpha": actual_lora.get("lora_alpha"),
        "actual_lora_dropout": actual_lora.get("lora_dropout"),
        "actual_target_modules": actual_lora.get("target_modules"),
        "actual_init_lora_weights": actual_lora.get("init_lora_weights"),
        "requested_input_config": config["input"],
        "actual_input_config": actual_input,
        "requested_training_objective": config["training"]["objective"],
        "actual_training_objective": actual_objective,
        "validation_start_step": config["validation"]["start_step"],
        "validation_interval_steps": config["validation"]["interval_steps"],
        "best_model_start_step": config["validation"]["best_model_start_step"],
        "resume_checkpoint_steps": config["checkpointing"]["resume_checkpoint_steps"],
        "overfitting": training_result.get("overfitting", {}),
        "best_adapter_path": summary.get(
            "best_adapter_path", config["derived"]["best_adapter_dir"]
        ),
        "final_adapter_path": summary.get(
            "final_adapter_path", config["derived"]["final_adapter_dir"]
        ),
        "resume_checkpoints": summary.get("resume_checkpoints", []),
        "training_peak_vram_allocated_gb": summary.get("peak_vram_allocated_gb"),
        "training_peak_vram_reserved_gb": summary.get("peak_vram_reserved_gb"),
        "metadata_json": config["paths"]["metadata_json"],
        "launch_resolved_config_yaml": config["paths"]["launch_resolved_config_yaml"],
        "runtime_resolved_config_json": config["paths"]["runtime_resolved_config_json"],
        "training_summary_json": config["paths"]["training_summary_json"],
        "validation_history_jsonl": config["paths"]["validation_history_jsonl"],
        "training_stdout_log": training_result.get("training_stdout_log"),
        "command": training_result.get("command", ""),
        "gpu_info": gpu_info,
        "dependency_status": dependency_status,
        "python_version": platform.python_version(),
        "notes": config.get("notes", {}),
    }


# ===========================================================================
# Output writers
# ===========================================================================

# Write the complete machine-readable experiment record.
def write_metadata_json(config: dict, metadata: dict) -> None:
    """Write the full metadata JSON file."""
    path = Path(config["paths"]["metadata_json"])
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


# Write a compact human-readable report for manual inspection.
def write_markdown_log(config: dict, metadata: dict) -> None:
    """Write a human-readable Markdown training log."""
    path = Path(config["paths"]["log_path"])
    overfit = metadata.get("overfitting", {})

    lines = [
        "# DF-04-02 LoRA Training Log",
        "",
        "## Status",
        "",
        f"- status: {metadata['status']}",
        f"- failure_reason: {metadata['failure_reason']}",
        f"- returncode: {metadata['returncode']}",
        "",
        "## Model Source",
        "",
        f"- model_hub_id: {metadata['model_hub_id']}",
        f"- requested_model_local_path: {metadata['requested_model_local_path']}",
        f"- actual_model_loaded_from: {metadata['actual_model_loaded_from']}",
        f"- actual_local_files_only: {metadata['actual_local_files_only']}",
        "",
        "## Actual LoRA Runtime Configuration",
        "",
        f"- actual_lora_rank: {metadata['actual_lora_rank']}",
        f"- actual_lora_alpha: {metadata['actual_lora_alpha']}",
        f"- actual_lora_dropout: {metadata['actual_lora_dropout']}",
        f"- actual_target_modules: {metadata['actual_target_modules']}",
        f"- actual_init_lora_weights: {metadata['actual_init_lora_weights']}",
        f"- trainable_params: {metadata['trainable_params']}",
        f"- total_params: {metadata['total_params']}",
        "",
        "## Training and Validation",
        "",
        f"- training_time_hours: {metadata['training_time_hours']}",
        f"- total_training_steps: {metadata['total_training_steps']}",
        f"- final_train_loss: {metadata['final_train_loss']}",
        f"- final_train_action_token_accuracy: {metadata['final_train_action_token_accuracy']}",
        f"- best_val_loss: {metadata['best_val_loss']}",
        f"- best_val_action_token_accuracy: {metadata['best_val_action_token_accuracy']}",
        f"- best_checkpoint_step: {metadata['best_checkpoint_step']}",
        f"- validation_start_step: {metadata['validation_start_step']}",
        f"- validation_interval_steps: {metadata['validation_interval_steps']}",
        f"- best_model_start_step: {metadata['best_model_start_step']}",
        f"- overfitting_checked: {overfit.get('overfitting_checked')}",
        f"- overfitting_detected: {overfit.get('overfitting_detected')}",
        f"- overfitting_reason: {overfit.get('overfitting_reason')}",
        "",
        "## Dataset and Normalization",
        "",
        f"- dataset_name: {metadata['dataset_name']}",
        f"- split_json: {metadata['split_json']}",
        f"- train_count: {metadata['train_count']}",
        f"- val_count: {metadata['val_count']}",
        f"- test_count: {metadata['test_count']}",
        f"- normalization_stats_json: {metadata['normalization_stats_json']}",
        f"- normalization_stats_source_requested: {metadata['normalization_stats_source_requested']}",
        f"- normalization_stats_source_actual: {metadata['normalization_stats_source_actual']}",
        "",
        "## Outputs",
        "",
        f"- best_adapter_path: {metadata['best_adapter_path']}",
        f"- final_adapter_path: {metadata['final_adapter_path']}",
        f"- resume_checkpoints: {metadata['resume_checkpoints']}",
        f"- runtime_resolved_config_json: {metadata['runtime_resolved_config_json']}",
        f"- training_summary_json: {metadata['training_summary_json']}",
        f"- validation_history_jsonl: {metadata['validation_history_jsonl']}",
        "",
        "## Command",
        "",
        "```bash",
        metadata.get("command", ""),
        "```",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# Write the comparison-friendly one-row summary used across experiment runs.
def write_summary_csv(config: dict, metadata: dict) -> None:
    """Write a one-row CSV summary for easy spreadsheet comparison."""
    path = Path(config["paths"]["summary_csv"])
    overfit = metadata.get("overfitting", {})

    row = {
        "run_id": metadata["run_id"],
        "branch_name": metadata["branch_name"],
        "git_commit_hash": metadata["git_commit_hash"],
        "status": metadata["status"],
        "failure_reason": metadata["failure_reason"],
        "model_hub_id": metadata["model_hub_id"],
        "actual_model_loaded_from": metadata["actual_model_loaded_from"],
        "dataset_name": metadata["dataset_name"],
        "train_count": metadata["train_count"],
        "val_count": metadata["val_count"],
        "test_count": metadata["test_count"],
        "training_time_hours": metadata["training_time_hours"],
        "total_training_steps": metadata["total_training_steps"],
        "final_train_loss": metadata["final_train_loss"],
        "final_train_action_token_accuracy": metadata[
            "final_train_action_token_accuracy"
        ],
        "best_val_loss": metadata["best_val_loss"],
        "best_val_action_token_accuracy": metadata[
            "best_val_action_token_accuracy"
        ],
        "best_checkpoint_step": metadata["best_checkpoint_step"],
        "overfitting_detected": overfit.get("overfitting_detected"),
        "overfitting_reason": overfit.get("overfitting_reason"),
        "per_device_batch_size": metadata["per_device_batch_size"],
        "effective_global_batch_size": metadata["effective_global_batch_size"],
        "learning_rate": metadata["learning_rate"],
        "actual_lora_rank": metadata["actual_lora_rank"],
        "actual_lora_alpha": metadata["actual_lora_alpha"],
        "actual_lora_dropout": metadata["actual_lora_dropout"],
        "actual_target_modules": json.dumps(
            metadata["actual_target_modules"], ensure_ascii=False
        ),
        "trainable_params": metadata["trainable_params"],
        "total_params": metadata["total_params"],
        "best_adapter_path": metadata["best_adapter_path"],
        "final_adapter_path": metadata["final_adapter_path"],
        "training_peak_vram_allocated_gb": metadata["training_peak_vram_allocated_gb"],
        "training_peak_vram_reserved_gb": metadata["training_peak_vram_reserved_gb"],
        "returncode": metadata["returncode"],
    }

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# ===========================================================================
# Entry point
# ===========================================================================

# Coordinate configuration, preflight validation, training, and final report generation.
def main() -> None:
    """
    Main orchestration entry point.

    Execution order:
      parse args -> load/validate config -> resolve paths -> seed ->
      prepare dirs -> preflight checks -> save launch config ->
      optionally train -> write metadata/log/csv -> exit
    """
    default_config = (
        Path(__file__).resolve().parent.parent / "configs" / DEFAULT_CONFIG_NAME
    )

    parser = argparse.ArgumentParser(
        description="Portable DF-04-02 LoRA training wrapper."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to train_config_lora_libero10_bs12_100k.yaml.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate model, split, stats, and source without training.",
    )
    args = parser.parse_args()

    # Resolve and validate all configuration before starting the training process.
    raw_config = load_config(args.config)
    validate_config(raw_config)
    config = resolve_config(raw_config, args.config)
    prepare_output_dirs(config)
    set_random_seed(int(config["run"]["random_seed"]))

    dependency_status = check_dependencies()
    gpu_info = get_gpu_info()
    git_commit_hash = get_git_commit_hash(Path(config["paths"]["project_root"]))

    preflight: Optional[dict] = None
    training_result: Dict[str, Any] = {}
    status = "failed"
    failure_reason = ""

    # Report generation remains outside this block so failures are still documented.
    try:
        preflight = run_preflight_checks(config)

        for required_package in ["torch", "transformers", "peft"]:
            if dependency_status.get(required_package) != "installed":
                raise ImportError(
                    f"{required_package} is not installed in the active environment."
                )

        command = build_finetune_command(config)
        save_launch_resolved_config(
            config=config,
            command=command,
            preflight=preflight,
            git_commit_hash=git_commit_hash,
        )

        if args.preflight_only:
            status = "preflight_passed"
        else:
            training_result = run_lora_training(config)

            if training_result.get("returncode") != 0:
                raise RuntimeError(
                    "LoRA finetune.py failed with returncode "
                    f"{training_result.get('returncode')}"
                )

            status = "training_finished"

    except Exception as error:
        failure_reason = str(error)
        status = "failed"

    # Always write final artifacts, including when preflight or training fails.
    metadata = build_metadata(
        config=config,
        dependency_status=dependency_status,
        gpu_info=gpu_info,
        preflight=preflight,
        status=status,
        training_result=training_result,
        failure_reason=failure_reason,
        git_commit_hash=git_commit_hash,
    )

    write_metadata_json(config, metadata)
    write_markdown_log(config, metadata)
    write_summary_csv(config, metadata)

    print(f"{TASK_ID} wrapper finished with status: {status}")
    print(f"Log: {config['paths']['log_path']}")
    print(f"Summary CSV: {config['paths']['summary_csv']}")
    print(f"Metadata JSON: {config['paths']['metadata_json']}")

    if status == "failed":
        print(f"Failure reason: {failure_reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
