from __future__ import annotations

# Notes:
# The following default paths are local paths on the author's cloud machine.
# They are recorded for experiment reproducibility and do not contain access credentials.
#
# DF-03-02 LIBERO Baseline 50-Sample Runner
#
# This runner is only used to validate the baseline inference pipeline and field recording.
# It is not used as a final method-comparison result.
#
# Important interpretation:
# - The 50 records are episode-level records selected from the fixed test split.
# - This runner enters each selected episode and selects one fixed step.
# - The selected image and language instruction are passed into the original OpenVLA baseline.
# - The output action is recorded for pipeline debugging only unless the LIBERO unnorm_key is confirmed.
# - Formal method evaluation will later use the full test split / full-step open-loop evaluation.
# - LIBERO simulation success rate is a separate rollout pipeline and is not measured here.
#
# Expected cloud dataset root:
# /root/autodl-tmp/datasets/openvla_modified_libero_rlds
#
# Expected input:
# results/splits/df_03_02_baseline_50_samples_seed42.json
#
# Expected outputs:
# results/tables/df_03_02_libero_baseline_50_metrics.csv
# logs/df_03_02_libero_baseline_50_log.md

import argparse
import csv
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# These CSV columns are kept stable so that later analysis scripts can read the file safely.
CSV_FIELDNAMES = [
    "timestamp",
    "run_id",
    "task_id",
    "branch_name",
    "git_commit_hash",
    "sample_id",
    "dataset_name",
    "dataset_split",
    "sub_dataset",
    "task_suite",
    "source_split",
    "eval_split",
    "episode_index",
    "version_dir",
    "dataset_dir",
    "fixed_step_policy",
    "fixed_step_index",
    "episode_num_steps",
    "instruction",
    "image_field",
    "language_field",
    "gt_action_json",
    "gt_action_dimension",
    "model_id",
    "unnorm_key",
    "unnorm_key_status",
    "action_values_official",
    "pipeline_debug_only",
    "device",
    "dtype",
    "python_version",
    "platform",
    "torch_version",
    "transformers_version",
    "tensorflow_datasets_version",
    "cuda_available",
    "cuda_version",
    "gpu_name",
    "gpu_total_vram_gb",
    "warmup_runs",
    "measured_runs",
    "processor_load_time_s",
    "model_load_time_s",
    "vram_before_loading_gb",
    "vram_after_loading_gb",
    "vram_before_inference_gb",
    "vram_after_inference_gb",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
    "step_load_time_ms",
    "preprocess_time_ms",
    "avg_latency_ms",
    "median_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "std_latency_ms",
    "postprocess_time_ms",
    "end_to_end_ms",
    "throughput_samples_per_sec",
    "raw_model_output",
    "parsed_action_json",
    "action_dimension",
    "action_parse_success",
    "action_range_valid",
    "action_range_error",
    "failure_reason",
    "inference_error_type",
    "inference_oom",
    "training_time_hours",
    "total_training_steps",
    "task_success_rate",
    "sim_environment",
    "notes",
]


# This function reads all command-line arguments used by the cloud runner.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DF-03-02 OpenVLA baseline inference on the fixed 50 LIBERO episodes."
    )

    # The manifest stores the fixed 50 episode-level samples selected from the test split.
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="results/splits/df_03_02_baseline_50_samples_seed42.json",
        help="Path to the fixed 50-episode baseline manifest JSON.",
    )

    # The dataset root points to the downloaded modified LIBERO RLDS dataset on the cloud disk.
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/root/autodl-tmp/datasets/openvla_modified_libero_rlds",
        help="Root path of the downloaded modified LIBERO RLDS dataset.",
    )

    # The model ID points to the original OpenVLA baseline model.
    parser.add_argument(
        "--model-id",
        type=str,
        default="openvla/openvla-7b",
        help="OpenVLA model ID on Hugging Face.",
    )

    # The unnormalization key is required because LIBERO-specific norm_stats were not found
    # in the original OpenVLA config during DF-03-02 inspection.
    parser.add_argument(
        "--unnorm-key",
        type=str,
        required=True,
        help=(
            "Action unnormalization key used by OpenVLA predict_action. "
            "For DF-03-02 pipeline debug, bridge_orig may be used only as an unconfirmed placeholder."
        ),
    )

    # This flag records whether the selected unnorm_key has been confirmed for LIBERO.
    parser.add_argument(
        "--unnorm-key-status",
        type=str,
        default="unconfirmed_for_libero",
        choices=["confirmed_for_libero", "unconfirmed_for_libero"],
        help="Whether the selected unnorm_key has been confirmed as correct for LIBERO action statistics.",
    )

    # The cloud GPU should normally be cuda:0.
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference, for example cuda:0 or cpu.",
    )

    # bfloat16 is the default because it is commonly used for large VLA inference on modern GPUs.
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype.",
    )

    # Warm-up runs are excluded from latency statistics.
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warm-up inference runs per sample.",
    )

    # Measured runs are used to calculate average, median, min, max, and standard deviation latency.
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help="Number of measured inference runs per sample.",
    )

    # The middle step is the default deterministic step policy for this DF-03-02 debug run.
    parser.add_argument(
        "--fixed-step-policy",
        type=str,
        default="middle",
        choices=["middle", "first", "last"],
        help="Policy for selecting one fixed step from each selected episode.",
    )

    # This field defines the expected OpenVLA action dimension.
    parser.add_argument(
        "--expected-action-dim",
        type=int,
        default=7,
        help="Expected OpenVLA action vector dimension.",
    )

    # This field catches NaN, Inf, and very large action values during pipeline validation.
    parser.add_argument(
        "--action-abs-limit",
        type=float,
        default=10.0,
        help="Maximum absolute value allowed for each parsed action element.",
    )

    # This optional limit is used for cloud smoke testing before running all 50 samples.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for debugging the runner.",
    )

    # Dry run checks arguments, manifest loading, and paths without loading OpenVLA.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and output path setup without loading OpenVLA or reading TFDS episodes.",
    )

    # The CSV output is the machine-readable DF-03-02 baseline validation result.
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/tables/df_03_02_libero_baseline_50_metrics.csv",
        help="CSV metrics output path.",
    )

    # The Markdown output is the human-readable DF-03-02 baseline validation log.
    parser.add_argument(
        "--output-log",
        type=str,
        default="logs/df_03_02_libero_baseline_50_log.md",
        help="Markdown log output path.",
    )

    # The run ID makes the generated outputs easier to trace later.
    parser.add_argument(
        "--run-id",
        type=str,
        default="df_03_02_libero_baseline_50_pipeline_debug",
        help="Unique run ID written into CSV and Markdown outputs.",
    )

    return parser.parse_args()


# This function imports optional packages only when they are needed.
def dynamic_import(package_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


# This function runs a small shell command and returns text output for metadata logging.
def run_command(command: List[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.stdout:
            return result.stdout.strip()

        if result.stderr:
            return result.stderr.strip()

        return "N/A"

    except Exception as error:
        return f"Command failed: {repr(error)}"


# This function reads the current Git branch name when available.
def get_git_branch() -> str:
    return run_command(["git", "branch", "--show-current"])


# This function reads the current Git commit hash when available.
def get_git_commit_hash() -> str:
    return run_command(["git", "rev-parse", "--short", "HEAD"])


# This function loads a JSON file from disk.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# This function writes rows to a CSV file using a stable field order.
def save_csv_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            normalized_row = {field: row.get(field, "") for field in CSV_FIELDNAMES}
            writer.writerow(normalized_row)


# This function writes a Markdown log file.
def write_markdown_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# This function prints a readable console section header.
def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# This function maps a dtype string to the matching PyTorch dtype object.
def get_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch_module.bfloat16

    if dtype_name == "float16":
        return torch_module.float16

    if dtype_name == "float32":
        return torch_module.float32

    return "auto"


# This function returns the installed package version when the module is available.
def get_module_version(module: Any) -> str:
    if module is None:
        return "not_available"

    return getattr(module, "__version__", "unknown")


# This function returns current and peak CUDA memory values in GB.
def get_cuda_memory_gb(torch_module: Any) -> Dict[str, float]:
    empty = {
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "max_allocated_gb": 0.0,
        "max_reserved_gb": 0.0,
        "total_vram_gb": 0.0,
    }

    if torch_module is None or not torch_module.cuda.is_available():
        return empty

    device_index = torch_module.cuda.current_device()
    total_vram_gb = torch_module.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)

    return {
        "allocated_gb": torch_module.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch_module.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch_module.cuda.max_memory_allocated() / (1024 ** 3),
        "max_reserved_gb": torch_module.cuda.max_memory_reserved() / (1024 ** 3),
        "total_vram_gb": total_vram_gb,
    }


# This function returns the active GPU name when CUDA is available.
def get_gpu_name(torch_module: Any) -> str:
    if torch_module is None or not torch_module.cuda.is_available():
        return "N/A"

    return torch_module.cuda.get_device_name(0)


# This function synchronizes CUDA timing when CUDA is available.
def synchronize_cuda(torch_module: Any) -> None:
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


# This function summarizes a list of latency values in milliseconds.
def summarize_latency_ms(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {
            "avg_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }

    return {
        "avg_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "std_ms": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
    }


# This function moves processor outputs to the selected device without casting integer token IDs to float.
def move_inputs_to_device(inputs: Dict[str, Any], device: str, dtype: Any) -> Dict[str, Any]:
    moved_inputs: Dict[str, Any] = {}

    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved_inputs[key] = value
            continue

        if hasattr(value, "is_floating_point") and value.is_floating_point():
            if dtype == "auto":
                moved_inputs[key] = value.to(device=device)
            else:
                moved_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            moved_inputs[key] = value.to(device=device)

    return moved_inputs


# This function converts TensorFlow tensors, NumPy arrays, bytes, and scalar values into Python values.
def to_python_value(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "numpy"):
        value = value.numpy()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return value


# This function converts a TFDS image value into a PIL image when PIL is available.
def to_pil_image(value: Any) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return value

    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(value)

    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    if array.dtype != np.uint8:
        array = array.clip(0, 255).astype(np.uint8)

    return Image.fromarray(array)


# This function searches nested dictionaries using multiple candidate paths.
def find_nested_value(data: Dict[str, Any], candidate_paths: Iterable[List[str]]) -> Any:
    for path in candidate_paths:
        current: Any = data
        found = True

        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break

            current = current[key]

        if found:
            return current

    return None


# This function builds the OpenVLA prompt format used for action prediction.
def make_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


# This function selects one deterministic step from an episode.
def select_fixed_step(steps: List[Dict[str, Any]], policy: str) -> Dict[str, Any]:
    if not steps:
        raise ValueError("Episode contains no steps.")

    if policy == "first":
        step_index = 0
    elif policy == "last":
        step_index = len(steps) - 1
    else:
        step_index = len(steps) // 2

    return {
        "step_index": step_index,
        "num_steps": len(steps),
        "step": steps[step_index],
    }


# This function converts a model or dataset action value into a flat Python float list.
def flatten_action_to_list(action: Any) -> List[float]:
    if action is None:
        return []

    try:
        import numpy as np
    except Exception:
        np = None

    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    elif hasattr(action, "numpy"):
        action = action.numpy()

    if np is not None:
        try:
            action_array = np.asarray(action).reshape(-1)
            return [float(value) for value in action_array.tolist()]
        except Exception:
            pass

    if isinstance(action, (list, tuple)):
        flattened: List[float] = []

        for item in action:
            flattened.extend(flatten_action_to_list(item))

        return flattened

    try:
        return [float(action)]
    except Exception:
        return []


# This function checks whether a parsed action has the expected dimension and valid finite values.
def check_action_range(
    parsed_action: Any,
    expected_dim: int = 7,
    abs_limit: float = 10.0,
) -> Dict[str, Any]:
    if not isinstance(parsed_action, (list, tuple)):
        return {
            "action_range_valid": False,
            "action_range_error": "parsed_action_is_not_list",
        }

    if len(parsed_action) != expected_dim:
        return {
            "action_range_valid": False,
            "action_range_error": f"expected_dim_{expected_dim}_got_{len(parsed_action)}",
        }

    for value in parsed_action:
        try:
            value_float = float(value)
        except Exception:
            return {
                "action_range_valid": False,
                "action_range_error": "action_value_not_numeric",
            }

        if not math.isfinite(value_float):
            return {
                "action_range_valid": False,
                "action_range_error": "action_contains_nan_or_inf",
            }

        if abs(value_float) > abs_limit:
            return {
                "action_range_valid": False,
                "action_range_error": f"action_exceeds_abs_limit_{abs_limit}",
            }

    return {
        "action_range_valid": True,
        "action_range_error": "",
    }


# This function parses the OpenVLA raw action output and validates its shape and numeric range.
def parse_action_output(
    action: Any,
    expected_dim: int,
    action_abs_limit: float,
) -> Dict[str, Any]:
    parsed_action = flatten_action_to_list(action)
    action_dimension = len(parsed_action)
    action_parse_success = action_dimension > 0

    range_info = check_action_range(
        parsed_action=parsed_action,
        expected_dim=expected_dim,
        abs_limit=action_abs_limit,
    )

    failure_reason = ""

    if not action_parse_success:
        failure_reason = "action output could not be parsed into a numeric list"
    elif not range_info["action_range_valid"]:
        failure_reason = range_info["action_range_error"]

    return {
        "raw_model_output": repr(action),
        "parsed_action_json": json.dumps(parsed_action, ensure_ascii=False, default=str),
        "action_dimension": action_dimension,
        "action_parse_success": action_parse_success,
        "action_range_valid": range_info["action_range_valid"],
        "action_range_error": range_info["action_range_error"],
        "failure_reason": failure_reason,
    }


# This function resolves the dataset version directory from either an absolute or relative manifest path.
def resolve_dataset_dir(sample: Dict[str, Any], dataset_root: Path) -> Path:
    version_dir_value = str(sample["version_dir"])
    version_dir = Path(version_dir_value)

    if version_dir.is_absolute():
        return version_dir

    candidate_with_sub_dataset = dataset_root / sample["sub_dataset"] / version_dir
    if candidate_with_sub_dataset.exists():
        return candidate_with_sub_dataset

    candidate_from_root = dataset_root / version_dir
    if candidate_from_root.exists():
        return candidate_from_root

    return candidate_with_sub_dataset


# This function extracts image, language instruction, and ground-truth action from a selected step.
def extract_step_fields(step: Dict[str, Any]) -> Dict[str, Any]:
    image = find_nested_value(
        step,
        [
            ["observation", "image"],
            ["observation", "rgb"],
            ["image"],
        ],
    )

    instruction = find_nested_value(
        step,
        [
            ["language_instruction"],
            ["natural_language_instruction"],
            ["instruction"],
            ["language"],
            ["observation", "language_instruction"],
            ["observation", "natural_language_instruction"],
            ["observation", "instruction"],
            ["observation", "language"],
        ],
    )

    gt_action = find_nested_value(
        step,
        [
            ["action"],
            ["actions"],
            ["target_action"],
        ],
    )

    if image is None:
        raise KeyError("No image field found in selected step.")

    if instruction is None:
        raise KeyError("No language instruction field found in selected step.")

    image_pil = to_pil_image(image)
    instruction_text = str(to_python_value(instruction))
    gt_action_list = flatten_action_to_list(gt_action)

    return {
        "image": image_pil,
        "instruction": instruction_text,
        "image_field": "observation.image",
        "language_field": "language_instruction",
        "gt_action_json": json.dumps(gt_action_list, ensure_ascii=False, default=str),
        "gt_action_dimension": len(gt_action_list),
    }


# This function loads one selected episode from TFDS and returns one deterministic selected step.
def load_episode_step(
    tfds_module: Any,
    sample: Dict[str, Any],
    dataset_root: Path,
    fixed_step_policy: str,
) -> Dict[str, Any]:
    dataset_dir = resolve_dataset_dir(sample=sample, dataset_root=dataset_root)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset version dir not found: {dataset_dir}")

    builder = tfds_module.builder_from_directory(str(dataset_dir))
    dataset = builder.as_dataset(split=sample["source_split"], shuffle_files=False)

    episode_index = int(sample["episode_index"])
    episode = next(iter(dataset.skip(episode_index).take(1)), None)

    if episode is None:
        raise IndexError(f"Episode index not found: {episode_index}")

    raw_steps = episode.get("steps") if isinstance(episode, dict) else None

    if raw_steps is None:
        raise KeyError("Episode has no steps field.")

    steps = list(raw_steps)
    selected = select_fixed_step(steps=steps, policy=fixed_step_policy)
    fields = extract_step_fields(selected["step"])

    return {
        "dataset_dir": str(dataset_dir),
        "fixed_step_policy": fixed_step_policy,
        "fixed_step_index": selected["step_index"],
        "episode_num_steps": selected["num_steps"],
        **fields,
    }


# This function runs warm-up and measured OpenVLA inference for one selected sample.
def run_single_sample(
    sample: Dict[str, Any],
    tfds_module: Any,
    processor: Any,
    model: Any,
    torch_module: Any,
    dataset_root: Path,
    device: str,
    dtype: Any,
    unnorm_key: str,
    fixed_step_policy: str,
    warmup_runs: int,
    measured_runs: int,
    expected_action_dim: int,
    action_abs_limit: float,
) -> Dict[str, Any]:
    if measured_runs <= 0:
        raise ValueError("measured_runs must be greater than 0.")

    sample_wall_start = time.perf_counter()

    step_load_start = time.perf_counter()
    step_record = load_episode_step(
        tfds_module=tfds_module,
        sample=sample,
        dataset_root=dataset_root,
        fixed_step_policy=fixed_step_policy,
    )
    step_load_time_ms = (time.perf_counter() - step_load_start) * 1000.0

    preprocess_start = time.perf_counter()
    prompt = make_prompt(step_record["instruction"])
    inputs = processor(prompt, step_record["image"], return_tensors="pt")
    inputs = move_inputs_to_device(dict(inputs), device=device, dtype=dtype)
    synchronize_cuda(torch_module)
    preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000.0

    # Warm-up inference is excluded from measured latency statistics.
    with torch_module.no_grad():
        for _ in range(warmup_runs):
            _ = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

    synchronize_cuda(torch_module)

    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    memory_before_inference = get_cuda_memory_gb(torch_module)

    latency_values_ms: List[float] = []
    last_action: Any = None

    # Measured inference runs are used for latency statistics.
    with torch_module.no_grad():
        for _ in range(measured_runs):
            synchronize_cuda(torch_module)
            inference_start = time.perf_counter()

            last_action = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

            synchronize_cuda(torch_module)
            latency_values_ms.append((time.perf_counter() - inference_start) * 1000.0)

    memory_after_inference = get_cuda_memory_gb(torch_module)

    postprocess_start = time.perf_counter()
    action_info = parse_action_output(
        action=last_action,
        expected_dim=expected_action_dim,
        action_abs_limit=action_abs_limit,
    )
    postprocess_time_ms = (time.perf_counter() - postprocess_start) * 1000.0

    latency_summary = summarize_latency_ms(latency_values_ms)

    end_to_end_ms = (
        preprocess_time_ms
        + latency_summary["avg_ms"]
        + postprocess_time_ms
    )

    throughput_samples_per_sec = 1000.0 / end_to_end_ms if end_to_end_ms > 0 else 0.0
    sample_wall_time_ms = (time.perf_counter() - sample_wall_start) * 1000.0

    return {
        "dataset_dir": step_record["dataset_dir"],
        "fixed_step_policy": step_record["fixed_step_policy"],
        "fixed_step_index": step_record["fixed_step_index"],
        "episode_num_steps": step_record["episode_num_steps"],
        "instruction": step_record["instruction"],
        "image_field": step_record["image_field"],
        "language_field": step_record["language_field"],
        "gt_action_json": step_record["gt_action_json"],
        "gt_action_dimension": step_record["gt_action_dimension"],
        "vram_before_inference_gb": round(memory_before_inference["allocated_gb"], 4),
        "vram_after_inference_gb": round(memory_after_inference["allocated_gb"], 4),
        "peak_vram_allocated_gb": round(memory_after_inference["max_allocated_gb"], 4),
        "peak_vram_reserved_gb": round(memory_after_inference["max_reserved_gb"], 4),
        "step_load_time_ms": round(step_load_time_ms, 4),
        "preprocess_time_ms": round(preprocess_time_ms, 4),
        "avg_latency_ms": round(latency_summary["avg_ms"], 4),
        "median_latency_ms": round(latency_summary["median_ms"], 4),
        "min_latency_ms": round(latency_summary["min_ms"], 4),
        "max_latency_ms": round(latency_summary["max_ms"], 4),
        "std_latency_ms": round(latency_summary["std_ms"], 4),
        "postprocess_time_ms": round(postprocess_time_ms, 4),
        "end_to_end_ms": round(end_to_end_ms, 4),
        "throughput_samples_per_sec": round(throughput_samples_per_sec, 6),
        "sample_wall_time_ms": round(sample_wall_time_ms, 4),
        "raw_model_output": action_info["raw_model_output"],
        "parsed_action_json": action_info["parsed_action_json"],
        "action_dimension": action_info["action_dimension"],
        "action_parse_success": action_info["action_parse_success"],
        "action_range_valid": action_info["action_range_valid"],
        "action_range_error": action_info["action_range_error"],
        "failure_reason": action_info["failure_reason"],
        "inference_error_type": "",
        "inference_oom": False,
    }


# This function returns a result structure when a sample fails.
def make_failure_result(error: Exception) -> Dict[str, Any]:
    error_text = repr(error)
    error_type = type(error).__name__
    inference_oom = "out of memory" in error_text.lower() or "cuda oom" in error_text.lower()

    return {
        "dataset_dir": "",
        "fixed_step_policy": "",
        "fixed_step_index": -1,
        "episode_num_steps": 0,
        "instruction": "",
        "image_field": "",
        "language_field": "",
        "gt_action_json": "[]",
        "gt_action_dimension": 0,
        "vram_before_inference_gb": 0.0,
        "vram_after_inference_gb": 0.0,
        "peak_vram_allocated_gb": 0.0,
        "peak_vram_reserved_gb": 0.0,
        "step_load_time_ms": 0.0,
        "preprocess_time_ms": 0.0,
        "avg_latency_ms": 0.0,
        "median_latency_ms": 0.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "std_latency_ms": 0.0,
        "postprocess_time_ms": 0.0,
        "end_to_end_ms": 0.0,
        "throughput_samples_per_sec": 0.0,
        "raw_model_output": "",
        "parsed_action_json": "[]",
        "action_dimension": 0,
        "action_parse_success": False,
        "action_range_valid": False,
        "action_range_error": "sample_failed_before_valid_action",
        "failure_reason": error_text,
        "inference_error_type": error_type,
        "inference_oom": inference_oom,
    }


# This function writes a dry-run log when the user only wants to validate paths and arguments.
def write_dry_run_log(
    output_log: Path,
    args: argparse.Namespace,
    manifest_json: Path,
    dataset_root: Path,
    sample_count: int,
    timestamp: str,
    git_branch: str,
    git_commit: str,
) -> None:
    lines = [
        "# DF-03-02 LIBERO Baseline 50-Sample Dry Run",
        "",
        "## Purpose",
        "",
        "This dry run validates argument parsing, manifest loading, and output path setup only.",
        "It does not load OpenVLA, read TFDS episodes, or run inference.",
        "",
        "## Configuration",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Git branch: `{git_branch}`",
        f"- Git commit: `{git_commit}`",
        f"- Manifest JSON: `{manifest_json}`",
        f"- Manifest exists: `{manifest_json.exists()}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Dataset root exists: `{dataset_root.exists()}`",
        f"- Model ID: `{args.model_id}`",
        f"- Unnorm key: `{args.unnorm_key}`",
        f"- Unnorm key status: `{args.unnorm_key_status}`",
        f"- Sample count: `{sample_count}`",
        "",
        "## Result",
        "",
        "Dry run finished successfully.",
        "",
    ]

    write_markdown_log(output_log, lines)


# This function writes the final human-readable Markdown log after a real inference run.
def write_final_log(
    output_log: Path,
    output_csv: Path,
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    timestamp: str,
    git_branch: str,
    git_commit: str,
    runtime_info: Dict[str, Any],
    processor_load_time_s: float,
    model_load_time_s: float,
    memory_before_loading: Dict[str, float],
    memory_after_loading: Dict[str, float],
) -> None:
    total_count = len(rows)
    parse_success_count = sum(1 for row in rows if row.get("action_parse_success") is True)
    range_valid_count = sum(1 for row in rows if row.get("action_range_valid") is True)
    error_count = sum(1 for row in rows if row.get("inference_error_type"))
    oom_count = sum(1 for row in rows if row.get("inference_oom") is True)

    valid_latency_values = [
        float(row["avg_latency_ms"])
        for row in rows
        if row.get("action_parse_success") is True and float(row.get("avg_latency_ms", 0.0)) > 0
    ]

    latency_summary = summarize_latency_ms(valid_latency_values)
    peak_vram_allocated = max([float(row.get("peak_vram_allocated_gb", 0.0)) for row in rows], default=0.0)
    peak_vram_reserved = max([float(row.get("peak_vram_reserved_gb", 0.0)) for row in rows], default=0.0)

    action_values_official = args.unnorm_key_status == "confirmed_for_libero"

    lines = [
        "# DF-03-02 LIBERO Baseline 50-Sample Run",
        "",
        "## Run Scope",
        "",
        "This 50-sample run is only used for DF-03-02 baseline pipeline and field-recording validation.",
        "It checks fixed step selection, image/instruction extraction, OpenVLA inference, action parsing, latency logging, and VRAM logging.",
        "",
        f"Action values use `{args.unnorm_key}` because LIBERO-specific norm_stats were not found in the original OpenVLA config during DF-03-02 inspection.",
        "Therefore action values are not official LIBERO evaluation results unless `unnorm_key_status` is confirmed.",
        "",
        "Formal method evaluation will use the full test split / full-step open-loop evaluation later.",
        "LIBERO simulation success rate is a separate rollout pipeline and is not measured in this run.",
        "",
        "## Configuration",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Git branch: `{git_branch}`",
        f"- Git commit: `{git_commit}`",
        f"- Manifest JSON: `{args.manifest_json}`",
        f"- Dataset root: `{args.dataset_root}`",
        f"- Model ID: `{args.model_id}`",
        f"- Unnorm key: `{args.unnorm_key}`",
        f"- Unnorm key status: `{args.unnorm_key_status}`",
        f"- Action values official: `{action_values_official}`",
        f"- Pipeline debug only: `{not action_values_official}`",
        f"- Device: `{args.device}`",
        f"- Dtype: `{args.dtype}`",
        f"- Warm-up runs: `{args.warmup_runs}`",
        f"- Measured runs: `{args.measured_runs}`",
        f"- Fixed step policy: `{args.fixed_step_policy}`",
        "",
        "## Runtime Environment",
        "",
        f"- Python version: `{runtime_info['python_version']}`",
        f"- Platform: `{runtime_info['platform']}`",
        f"- Torch version: `{runtime_info['torch_version']}`",
        f"- Transformers version: `{runtime_info['transformers_version']}`",
        f"- TensorFlow Datasets version: `{runtime_info['tensorflow_datasets_version']}`",
        f"- CUDA available: `{runtime_info['cuda_available']}`",
        f"- CUDA version: `{runtime_info['cuda_version']}`",
        f"- GPU name: `{runtime_info['gpu_name']}`",
        f"- GPU total VRAM GB: `{runtime_info['gpu_total_vram_gb']:.4f}`",
        "",
        "## Model Loading",
        "",
        f"- Processor load time seconds: `{processor_load_time_s:.4f}`",
        f"- Model load time seconds: `{model_load_time_s:.4f}`",
        f"- VRAM before loading GB: `{memory_before_loading['allocated_gb']:.4f}`",
        f"- VRAM after loading GB: `{memory_after_loading['allocated_gb']:.4f}`",
        "",
        "## Result Summary",
        "",
        f"- Total samples: `{total_count}`",
        f"- Action parse success count: `{parse_success_count}`",
        f"- Action parse success rate: `{parse_success_count / total_count if total_count else 0.0:.4f}`",
        f"- Action range valid count: `{range_valid_count}`",
        f"- Action range valid rate: `{range_valid_count / total_count if total_count else 0.0:.4f}`",
        f"- Inference error count: `{error_count}`",
        f"- Inference OOM count: `{oom_count}`",
        f"- Overall average latency from sample averages: `{latency_summary['avg_ms']:.4f} ms`",
        f"- Overall median latency from sample averages: `{latency_summary['median_ms']:.4f} ms`",
        f"- Peak VRAM allocated: `{peak_vram_allocated:.4f} GB`",
        f"- Peak VRAM reserved: `{peak_vram_reserved:.4f} GB`",
        "",
        "## Output Files",
        "",
        f"- CSV metrics: `{output_csv}`",
        f"- Markdown log: `{output_log}`",
        "",
        "## Important Interpretation",
        "",
        "This run is pipeline and field-recording validation only.",
        "It must not be used as the final method-comparison result.",
        "Formal evaluation will be run later on the full test split / full-step open-loop evaluation.",
        "",
    ]

    write_markdown_log(output_log, lines)


# This function runs the whole DF-03-02 baseline validation flow.
def main() -> None:
    args = parse_args()

    manifest_json = Path(args.manifest_json)
    dataset_root = Path(args.dataset_root)
    output_csv = Path(args.output_csv)
    output_log = Path(args.output_log)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_branch = get_git_branch()
    git_commit = get_git_commit_hash()

    action_values_official = args.unnorm_key_status == "confirmed_for_libero"

    print_section("DF-03-02 LIBERO Baseline 50-Sample Runner")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest exists: {manifest_json.exists()}")
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset root exists: {dataset_root.exists()}")
    print(f"Model ID: {args.model_id}")
    print(f"Unnorm key: {args.unnorm_key}")
    print(f"Unnorm key status: {args.unnorm_key_status}")
    print(f"Action values official: {action_values_official}")
    print(f"Pipeline debug only: {not action_values_official}")

    if not manifest_json.exists():
        raise FileNotFoundError(f"Manifest JSON was not found: {manifest_json}")

    manifest = load_json(manifest_json)
    samples = list(manifest.get("samples", []))

    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        raise ValueError("Manifest does not contain any samples after applying the optional limit.")

    if args.dry_run:
        write_dry_run_log(
            output_log=output_log,
            args=args,
            manifest_json=manifest_json,
            dataset_root=dataset_root,
            sample_count=len(samples),
            timestamp=timestamp,
            git_branch=git_branch,
            git_commit=git_commit,
        )

        print("Dry run finished.")
        print(f"Markdown log saved to: {output_log}")
        return

    torch_module = dynamic_import("torch")
    transformers_module = dynamic_import("transformers")
    tfds_module = dynamic_import("tensorflow_datasets")

    if torch_module is None or transformers_module is None or tfds_module is None:
        missing = [
            name
            for name, module in [
                ("torch", torch_module),
                ("transformers", transformers_module),
                ("tensorflow_datasets", tfds_module),
            ]
            if module is None
        ]
        raise ImportError(f"Missing required cloud/HPC packages: {', '.join(missing)}")

    dtype = get_torch_dtype(torch_module, args.dtype)

    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    memory_before_loading = get_cuda_memory_gb(torch_module)

    print_section("Loading OpenVLA")
    processor_load_start = time.perf_counter()

    processor = transformers_module.AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    processor_load_time_s = time.perf_counter() - processor_load_start

    model_load_start = time.perf_counter()

    model = transformers_module.AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = model.to(args.device)
    model.eval()
    synchronize_cuda(torch_module)

    model_load_time_s = time.perf_counter() - model_load_start
    memory_after_loading = get_cuda_memory_gb(torch_module)

    gpu_name = get_gpu_name(torch_module)

    runtime_info = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": get_module_version(torch_module),
        "transformers_version": get_module_version(transformers_module),
        "tensorflow_datasets_version": get_module_version(tfds_module),
        "cuda_available": torch_module.cuda.is_available(),
        "cuda_version": getattr(torch_module.version, "cuda", "unknown"),
        "gpu_name": gpu_name,
        "gpu_total_vram_gb": memory_after_loading["total_vram_gb"],
    }

    print(f"GPU: {gpu_name}")
    print(f"Processor loaded in {processor_load_time_s:.2f} seconds.")
    print(f"Model loaded in {model_load_time_s:.2f} seconds.")

    result_rows: List[Dict[str, Any]] = []

    print_section("Running Samples")

    for sample_index, sample in enumerate(samples, start=1):
        print(f"Running sample {sample_index}/{len(samples)}: {sample.get('sample_id', 'unknown_sample')}")

        try:
            sample_result = run_single_sample(
                sample=sample,
                tfds_module=tfds_module,
                processor=processor,
                model=model,
                torch_module=torch_module,
                dataset_root=dataset_root,
                device=args.device,
                dtype=dtype,
                unnorm_key=args.unnorm_key,
                fixed_step_policy=args.fixed_step_policy,
                warmup_runs=args.warmup_runs,
                measured_runs=args.measured_runs,
                expected_action_dim=args.expected_action_dim,
                action_abs_limit=args.action_abs_limit,
            )

        except Exception as error:
            sample_result = make_failure_result(error)

        row = {
            "timestamp": timestamp,
            "run_id": args.run_id,
            "task_id": manifest.get("task_id", "DF-03-02"),
            "branch_name": git_branch,
            "git_commit_hash": git_commit,
            "sample_id": sample.get("sample_id", ""),
            "dataset_name": sample.get("dataset_name", manifest.get("dataset_name", "")),
            "dataset_split": sample.get("eval_split", ""),
            "sub_dataset": sample.get("sub_dataset", ""),
            "task_suite": sample.get("task_suite", ""),
            "source_split": sample.get("source_split", ""),
            "eval_split": sample.get("eval_split", ""),
            "episode_index": sample.get("episode_index", ""),
            "version_dir": sample.get("version_dir", ""),
            "model_id": args.model_id,
            "unnorm_key": args.unnorm_key,
            "unnorm_key_status": args.unnorm_key_status,
            "action_values_official": action_values_official,
            "pipeline_debug_only": not action_values_official,
            "device": args.device,
            "dtype": args.dtype,
            "python_version": runtime_info["python_version"],
            "platform": runtime_info["platform"],
            "torch_version": runtime_info["torch_version"],
            "transformers_version": runtime_info["transformers_version"],
            "tensorflow_datasets_version": runtime_info["tensorflow_datasets_version"],
            "cuda_available": runtime_info["cuda_available"],
            "cuda_version": runtime_info["cuda_version"],
            "gpu_name": runtime_info["gpu_name"],
            "gpu_total_vram_gb": round(runtime_info["gpu_total_vram_gb"], 4),
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "processor_load_time_s": round(processor_load_time_s, 4),
            "model_load_time_s": round(model_load_time_s, 4),
            "vram_before_loading_gb": round(memory_before_loading["allocated_gb"], 4),
            "vram_after_loading_gb": round(memory_after_loading["allocated_gb"], 4),
            "training_time_hours": "not_applicable",
            "total_training_steps": "not_applicable",
            "task_success_rate": "not_applicable",
            "sim_environment": "not_applicable",
            "notes": (
                "DF-03-02 50-sample run is pipeline and field-recording validation only; "
                "not final method-comparison data."
            ),
            **sample_result,
        }

        result_rows.append(row)

        print(f"Average latency: {row['avg_latency_ms']} ms")
        print(f"Action parse success: {row['action_parse_success']}")
        print(f"Action range valid: {row['action_range_valid']}")
        print(f"Peak VRAM allocated: {row['peak_vram_allocated_gb']} GB")

    save_csv_rows(output_csv, result_rows)

    write_final_log(
        output_log=output_log,
        output_csv=output_csv,
        args=args,
        rows=result_rows,
        timestamp=timestamp,
        git_branch=git_branch,
        git_commit=git_commit,
        runtime_info=runtime_info,
        processor_load_time_s=processor_load_time_s,
        model_load_time_s=model_load_time_s,
        memory_before_loading=memory_before_loading,
        memory_after_loading=memory_after_loading,
    )

    print()
    print("Baseline 50-sample run finished.")
    print(f"CSV saved to: {output_csv}")
    print(f"Markdown log saved to: {output_log}")


if __name__ == "__main__":
    main()