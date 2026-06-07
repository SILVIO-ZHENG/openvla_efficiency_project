"""
中文：批量运行 OpenVLA baseline smoke test，一次加载模型，循环测试多张图片。
English: Run OpenVLA baseline smoke test in batch mode by loading the model once and testing multiple images.

中文说明：
- 这个脚本用于 DF-02-03。
- 目标是云端 GPU / HPC smoke test，不是最终论文正式 evaluation。
- 本地可以用 --dry-run 检查 CSV 和路径逻辑。
- 云端运行时会输出 latency、VRAM、sample action、CSV 和 Markdown log。
- 不要在本地强行安装重型依赖；真正 OpenVLA 运行放到云端 GPU / HPC。

English note:
- This script is used for DF-02-03.
- The goal is a cloud GPU / HPC smoke test, not the final dissertation evaluation.
- Local --dry-run only checks CSV and path logic.
- Cloud execution records latency, VRAM, sample actions, CSV, and Markdown log.
- Heavy OpenVLA execution should be done on cloud GPU / HPC.
"""

import argparse
import csv
import importlib
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def print_section(title: str) -> None:
    """中文：打印分隔标题。English: Print a section title."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def dynamic_import(package_name: str) -> Optional[Any]:
    """
    中文：动态导入依赖，避免本地没有 torch / transformers / pillow 时直接报错。
    English: Dynamically import dependencies to avoid crashing when torch / transformers / pillow are missing locally.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


def run_command(command: List[str]) -> str:
    """
    中文：运行系统命令并返回输出。
    English: Run a system command and return its output.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.stdout:
            return result.stdout.strip()

        if result.stderr:
            return result.stderr.strip()

        return "N/A"

    except Exception as error:
        return f"Command failed: {repr(error)}"


def get_git_commit_hash() -> str:
    """中文：获取当前 Git commit hash。English: Get current Git commit hash."""
    return run_command(["git", "rev-parse", "--short", "HEAD"])


def get_git_branch() -> str:
    """中文：获取当前 Git 分支。English: Get current Git branch."""
    return run_command(["git", "branch", "--show-current"])


def get_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """
    中文：根据字符串选择 torch dtype。
    English: Select torch dtype according to a string name.
    """
    if dtype_name == "bfloat16":
        return torch_module.bfloat16

    if dtype_name == "float16":
        return torch_module.float16

    if dtype_name == "float32":
        return torch_module.float32

    return "auto"


def get_cuda_memory_gb(torch_module: Any) -> Dict[str, float]:
    """
    中文：获取 CUDA 显存使用情况。
    English: Get CUDA memory usage.
    """
    if torch_module is None:
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
            "total_vram_gb": 0.0,
        }

    if not torch_module.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
            "total_vram_gb": 0.0,
        }

    device_index = torch_module.cuda.current_device()
    total_vram_gb = torch_module.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)

    return {
        "allocated_gb": torch_module.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch_module.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch_module.cuda.max_memory_allocated() / (1024 ** 3),
        "max_reserved_gb": torch_module.cuda.max_memory_reserved() / (1024 ** 3),
        "total_vram_gb": total_vram_gb,
    }


def get_gpu_name(torch_module: Any) -> str:
    """
    中文：获取 GPU 名称。
    English: Get GPU name.
    """
    if torch_module is None or not torch_module.cuda.is_available():
        return "N/A"

    return torch_module.cuda.get_device_name(0)


def read_instruction_csv(csv_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    中文：读取包含 image_path 和 instruction 的 CSV。
    English: Read a CSV file containing image_path and instruction columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    samples: List[Dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"image_path", "instruction"}
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV file is missing columns: {missing_columns}")

        for row_index, row in enumerate(reader):
            image_path = str(row.get("image_path", "")).strip()
            instruction = str(row.get("instruction", "")).strip()
            note = str(row.get("note", "")).strip()

            if not image_path or not instruction:
                continue

            samples.append(
                {
                    "sample_id": f"sample_{row_index + 1:03d}",
                    "image_path": image_path,
                    "instruction": instruction,
                    "note": note,
                }
            )

            if limit is not None and len(samples) >= limit:
                break

    return samples


def append_markdown_log(log_path: Path, lines: List[str]) -> None:
    """
    中文：追加写入 Markdown 日志。
    English: Append lines to a Markdown log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n\n")


def save_csv_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    中文：保存多行结果到 CSV。
    English: Save multiple result rows to a CSV file.
    """
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_latency_ms(values_ms: List[float]) -> Dict[str, float]:
    """
    中文：统计 latency 的平均值、中位数、最小值、最大值和标准差。
    English: Summarise latency using mean, median, min, max, and standard deviation.
    """
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


def load_image(image_path: Path) -> Any:
    """
    中文：加载输入图片。
    English: Load an input image.
    """
    pil_image_module = dynamic_import("PIL.Image")

    if pil_image_module is None:
        raise ImportError("PIL is not installed. Please install pillow on cloud GPU / HPC.")

    return pil_image_module.open(image_path).convert("RGB")


def move_inputs_to_device(inputs: Dict[str, Any], device: str, dtype: Any) -> Dict[str, Any]:
    """
    中文：把 processor 输出移动到指定设备。
    English: Move processor outputs to the target device.

    中文：
    - 浮点 tensor 使用 dtype。
    - input_ids 这类整数 tensor 保持原 dtype。
    English:
    - Floating tensors use the selected dtype.
    - Integer tensors such as input_ids keep their original dtype.
    """
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


def parse_action_output(action: Any) -> Dict[str, Any]:
    """
    中文：解析模型输出 action。
    English: Parse model action output.
    """
    if action is None:
        return {
            "parsed_action": [],
            "action_dimension": 0,
            "action_parse_success": False,
            "action_range_valid": False,
            "failure_reason": "no output",
        }

    if hasattr(action, "tolist"):
        parsed_action = action.tolist()
    else:
        parsed_action = action

    if isinstance(parsed_action, (list, tuple)):
        action_dimension = len(parsed_action)
    else:
        action_dimension = 0

    # 中文：这里只做非常轻量的检查，不假设具体机器人动作范围。
    # English: Only a lightweight check is used here because the exact robot action range is not assumed.
    action_range_valid = action_dimension > 0

    return {
        "parsed_action": parsed_action,
        "action_dimension": action_dimension,
        "action_parse_success": True,
        "action_range_valid": action_range_valid,
        "failure_reason": "",
    }


def make_prompt(instruction: str) -> str:
    """
    中文：构造 OpenVLA 常用 prompt。
    English: Build the common OpenVLA prompt.
    """
    return f"In: What action should the robot take to {instruction}?\nOut:"


def run_single_sample(
    sample: Dict[str, str],
    processor: Any,
    model: Any,
    torch_module: Any,
    device: str,
    dtype: Any,
    unnorm_key: str,
    warmup_runs: int,
    num_runs: int,
) -> Dict[str, Any]:
    """
    中文：对单个样本运行 baseline inference。
    English: Run baseline inference for a single sample.
    """
    image_path = Path(sample["image_path"])
    instruction = sample["instruction"]

    if not image_path.exists():
        return {
            "sample_id": sample["sample_id"],
            "image_path": str(image_path),
            "instruction": instruction,
            "note": sample.get("note", ""),
            "preprocess_time_ms": 0.0,
            "avg_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "end_to_end_time_ms": 0.0,
            "action_dimension": 0,
            "action_parse_success": False,
            "action_range_valid": False,
            "parsed_action_json": "[]",
            "failure_reason": "image file not found",
        }

    end_to_end_start = time.perf_counter()

    preprocess_start = time.perf_counter()

    image = load_image(image_path)
    prompt = make_prompt(instruction)

    inputs = processor(
        prompt,
        image,
        return_tensors="pt",
    )

    inputs = move_inputs_to_device(
        inputs=dict(inputs),
        device=device,
        dtype=dtype,
    )

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000

    with torch_module.no_grad():
        for _ in range(warmup_runs):
            _ = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

    latency_values_ms: List[float] = []
    last_action: Any = None

    with torch_module.no_grad():
        for _ in range(num_runs):
            if torch_module.cuda.is_available():
                torch_module.cuda.synchronize()

            inference_start = time.perf_counter()

            action = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

            if torch_module.cuda.is_available():
                torch_module.cuda.synchronize()

            inference_time_ms = (time.perf_counter() - inference_start) * 1000
            latency_values_ms.append(inference_time_ms)
            last_action = action

    end_to_end_time_ms = (time.perf_counter() - end_to_end_start) * 1000

    latency_summary = summarise_latency_ms(latency_values_ms)
    action_info = parse_action_output(last_action)

    return {
        "sample_id": sample["sample_id"],
        "image_path": str(image_path),
        "instruction": instruction,
        "note": sample.get("note", ""),
        "preprocess_time_ms": round(preprocess_time_ms, 4),
        "avg_latency_ms": round(latency_summary["avg_ms"], 4),
        "median_latency_ms": round(latency_summary["median_ms"], 4),
        "min_latency_ms": round(latency_summary["min_ms"], 4),
        "max_latency_ms": round(latency_summary["max_ms"], 4),
        "std_latency_ms": round(latency_summary["std_ms"], 4),
        "end_to_end_time_ms": round(end_to_end_time_ms, 4),
        "action_dimension": action_info["action_dimension"],
        "action_parse_success": action_info["action_parse_success"],
        "action_range_valid": action_info["action_range_valid"],
        "parsed_action_json": json.dumps(action_info["parsed_action"], ensure_ascii=False, default=str),
        "failure_reason": action_info["failure_reason"],
    }


def main() -> None:
    """中文：主函数。English: Main function."""
    parser = argparse.ArgumentParser(description="Run OpenVLA baseline batch smoke test.")

    parser.add_argument(
        "--instruction-csv",
        type=str,
        default="test_instructions.csv",
        help="CSV file with image_path and instruction columns.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="openvla/openvla-7b",
        help="OpenVLA model id on Hugging Face.",
    )

    parser.add_argument(
        "--unnorm-key",
        type=str,
        default="bridge_orig",
        help="Action unnormalization key used by OpenVLA.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference, for example cuda:0 or cpu.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warm-up inference runs per sample.",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of measured inference runs per sample.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of samples to run.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check CSV and paths without loading OpenVLA.",
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/df_02_03_cloud_smoke_test_log.md",
        help="Markdown log output path.",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/tables/df_02_03_cloud_smoke_test_metrics.csv",
        help="CSV metrics output path.",
    )

    args = parser.parse_args()

    instruction_csv = Path(args.instruction_csv)
    log_path = Path(args.log_path)
    csv_path = Path(args.csv_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_branch = get_git_branch()
    git_commit = get_git_commit_hash()

    print_section("DF-02-03 OpenVLA Batch Baseline Smoke Test / 批量 Baseline 跑通测试")

    samples = read_instruction_csv(instruction_csv, limit=args.limit)

    print(f"CSV path: {instruction_csv}")
    print(f"Loaded samples: {len(samples)}")

    log_lines = [
        "# DF-02-03 Cloud Baseline Smoke Test Batch Run",
        "",
        "## Timestamp",
        f"- {timestamp}",
        "",
        "## Git Information",
        f"- Branch: `{git_branch}`",
        f"- Commit: `{git_commit}`",
        "",
        "## Configuration",
        f"- Model ID: `{args.model_id}`",
        f"- Device: `{args.device}`",
        f"- Dtype: `{args.dtype}`",
        f"- Instruction CSV: `{instruction_csv}`",
        f"- Warm-up runs per sample: `{args.warmup_runs}`",
        f"- Measured runs per sample: `{args.num_runs}`",
        f"- Sample limit: `{args.limit}`",
        f"- Dry run: `{args.dry_run}`",
        "",
        "## Sample List",
    ]

    for sample in samples:
        image_exists = Path(sample["image_path"]).exists()
        print(f"{sample['sample_id']}: {sample['image_path']} | exists={image_exists} | {sample['instruction']}")
        log_lines.append(
            f"- `{sample['sample_id']}` | `{sample['image_path']}` | exists=`{image_exists}` | instruction=`{sample['instruction']}`"
        )

    log_lines.append("")

    if args.dry_run:
        print_section("Dry Run Finished / Dry Run 完成")
        print("No model was loaded.")
        print("This only checks whether the CSV can be read and image paths are visible.")

        log_lines.extend(
            [
                "## Dry Run Result",
                "- No model was loaded.",
                "- CSV reading succeeded.",
                "- Image path visibility was checked.",
                "",
            ]
        )

        append_markdown_log(log_path, log_lines)
        return

    torch_module = dynamic_import("torch")
    transformers_module = dynamic_import("transformers")

    if torch_module is None or transformers_module is None:
        print("ERROR: PyTorch or transformers is not installed.")
        print("Install them on the cloud GPU / HPC machine before running real baseline inference.")

        log_lines.extend(
            [
                "## Result",
                "- Failed before model loading.",
                "- PyTorch or transformers is not installed.",
                "",
            ]
        )

        append_markdown_log(log_path, log_lines)
        return

    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    dtype = get_torch_dtype(torch_module, args.dtype)

    memory_before_loading = get_cuda_memory_gb(torch_module)
    gpu_name = get_gpu_name(torch_module)

    print_section("Loading OpenVLA Once / 只加载一次 OpenVLA")

    load_start = time.perf_counter()

    processor = transformers_module.AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    model = transformers_module.AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = model.to(args.device)
    model.eval()

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    model_load_time_seconds = time.perf_counter() - load_start
    memory_after_loading = get_cuda_memory_gb(torch_module)

    print(f"GPU: {gpu_name}")
    print(f"Model loaded in {model_load_time_seconds:.2f} seconds.")
    print(f"VRAM after loading: {memory_after_loading['allocated_gb']:.4f} GB")

    result_rows: List[Dict[str, Any]] = []

    print_section("Running Batch Inference / 开始批量推理")

    for sample_index, sample in enumerate(samples):
        print(f"\nRunning {sample_index + 1}/{len(samples)}: {sample['sample_id']}")

        sample_result = run_single_sample(
            sample=sample,
            processor=processor,
            model=model,
            torch_module=torch_module,
            device=args.device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
        )

        memory_after_sample = get_cuda_memory_gb(torch_module)

        row = {
            "timestamp": timestamp,
            "branch": git_branch,
            "commit": git_commit,
            "model_id": args.model_id,
            "gpu_name": gpu_name,
            "device": args.device,
            "dtype": args.dtype,
            "warmup_runs": args.warmup_runs,
            "num_runs": args.num_runs,
            "model_load_time_seconds": round(model_load_time_seconds, 4),
            "vram_before_loading_gb": round(memory_before_loading["allocated_gb"], 4),
            "vram_after_loading_gb": round(memory_after_loading["allocated_gb"], 4),
            "vram_after_sample_gb": round(memory_after_sample["allocated_gb"], 4),
            "peak_vram_allocated_gb": round(memory_after_sample["max_allocated_gb"], 4),
            "peak_vram_reserved_gb": round(memory_after_sample["max_reserved_gb"], 4),
            "gpu_total_vram_gb": round(memory_after_sample["total_vram_gb"], 4),
            **sample_result,
        }

        result_rows.append(row)

        print(f"Average latency: {row['avg_latency_ms']} ms")
        print(f"Action parse success: {row['action_parse_success']}")
        print(f"Peak VRAM allocated: {row['peak_vram_allocated_gb']} GB")

    save_csv_rows(csv_path, result_rows)

    all_avg_latencies = [
        float(row["avg_latency_ms"])
        for row in result_rows
        if row.get("action_parse_success") is True
    ]
    overall_latency = summarise_latency_ms(all_avg_latencies)

    success_count = sum(1 for row in result_rows if row.get("action_parse_success") is True)
    total_count = len(result_rows)
    parse_success_rate = success_count / total_count if total_count > 0 else 0.0

    final_memory = get_cuda_memory_gb(torch_module)

    log_lines.extend(
        [
            "## Cloud Run Result",
            f"- GPU: `{gpu_name}`",
            f"- Model loaded: `True`",
            f"- Model loading time: `{model_load_time_seconds:.4f} seconds`",
            f"- Total samples: `{total_count}`",
            f"- Action parse success count: `{success_count}`",
            f"- Action parse success rate: `{parse_success_rate:.4f}`",
            f"- Overall average latency from sample averages: `{overall_latency['avg_ms']:.4f} ms`",
            f"- Overall median latency from sample averages: `{overall_latency['median_ms']:.4f} ms`",
            f"- Peak VRAM allocated: `{final_memory['max_allocated_gb']:.4f} GB`",
            f"- Peak VRAM reserved: `{final_memory['max_reserved_gb']:.4f} GB`",
            f"- CSV saved to: `{csv_path}`",
            "",
            "## Notes",
            "- This is a smoke test, not the final dissertation evaluation.",
            "- Formal comparison should be re-run after the fixed evaluation split is prepared.",
            "",
        ]
    )

    append_markdown_log(log_path, log_lines)

    print_section("Batch Smoke Test Finished / 批量跑通测试完成")
    print(f"Total samples: {total_count}")
    print(f"Action parse success rate: {parse_success_rate:.4f}")
    print(f"Overall avg latency: {overall_latency['avg_ms']:.4f} ms")
    print(f"Peak VRAM allocated: {final_memory['max_allocated_gb']:.4f} GB")
    print(f"CSV saved to: {csv_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()