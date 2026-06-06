"""
中文：运行 OpenVLA baseline 推理，并记录 latency、VRAM 和 sample action output。
English: Run OpenVLA baseline inference and record latency, VRAM, and sample action output.

中文说明：
- 本地电脑主要用于代码检查、Git 管理和 dry run。
- 真正的 OpenVLA baseline inference 应该在云端 GPU / 学校 HPC 上运行。
- 如果本地没有 torch / transformers / pillow，这是正常的。
- 本脚本会在云端运行时输出 baseline 结果，用来后面和 LoRA、QLoRA、quantization 做对比。

English note:
- The local machine is mainly used for code checking, Git management, and dry run.
- Real OpenVLA baseline inference should run on cloud GPU / university HPC.
- It is normal if torch / transformers / pillow are not installed locally.
- This script records baseline results for later comparison with LoRA, QLoRA, and quantization.
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
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def dynamic_import(package_name: str) -> Optional[Any]:
    """
    中文：动态导入依赖，避免本地没有重型依赖时报错。
    English: Dynamically import dependencies to avoid local missing-package errors.
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


def get_cuda_memory_gb(torch_module: Any) -> Dict[str, float]:
    """
    中文：获取当前 CUDA 显存使用情况。
    English: Get current CUDA memory usage.
    """
    if torch_module is None:
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
        }

    if not torch_module.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
        }

    return {
        "allocated_gb": torch_module.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch_module.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch_module.cuda.max_memory_allocated() / (1024 ** 3),
        "max_reserved_gb": torch_module.cuda.max_memory_reserved() / (1024 ** 3),
    }


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


def append_markdown_log(log_path: Path, lines: List[str]) -> None:
    """
    中文：追加写入 markdown 日志。
    English: Append lines to a markdown log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n\n")


def save_csv_result(csv_path: Path, row: Dict[str, Any]) -> None:
    """
    中文：保存 baseline 结果到 CSV，方便后续论文表格使用。
    English: Save baseline results to CSV for later dissertation tables.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def summarise_latency(latency_list: List[float]) -> Dict[str, float]:
    """
    中文：计算 latency 的平均值、中位数、最小值、最大值和标准差。
    English: Calculate average, median, minimum, maximum, and standard deviation of latency.
    """
    if not latency_list:
        # 中文：没有 latency 数据时返回 0，避免 round() 处理 None 时被 Pylance 报错。
        # English: Return 0 when no latency data exists to avoid Pylance errors with round(None).
        return {
            "avg_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }

    # 中文：脚本内部计时单位是秒，这里转换成毫秒。
    # English: Internal timing is in seconds, so it is converted to milliseconds here.
    latency_ms = [value * 1000 for value in latency_list]

    return {
        "avg_ms": statistics.mean(latency_ms),
        "median_ms": statistics.median(latency_ms),
        "min_ms": min(latency_ms),
        "max_ms": max(latency_ms),
        "std_ms": statistics.stdev(latency_ms) if len(latency_ms) > 1 else 0.0,
    }


def load_image(image_path: str) -> Any:
    """
    中文：加载输入图片。
    English: Load the input image.
    """
    pil_image_module = dynamic_import("PIL.Image")

    if pil_image_module is None:
        raise ImportError("PIL is not installed. Please install pillow on cloud GPU / HPC.")

    return pil_image_module.open(image_path).convert("RGB")


def move_inputs_to_device(inputs: Dict[str, Any], torch_module: Any, device: str, dtype: Any) -> Dict[str, Any]:
    """
    中文：把 processor 输出移动到指定设备。
    English: Move processor outputs to the target device.

    中文：
    - float tensor 使用指定 dtype。
    - input_ids 这类整数 tensor 不改 dtype。
    English:
    - Float tensors use the selected dtype.
    - Integer tensors such as input_ids keep their original dtype.
    """
    moved_inputs: Dict[str, Any] = {}

    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved_inputs[key] = value
            continue

        if hasattr(value, "is_floating_point") and value.is_floating_point():
            moved_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            moved_inputs[key] = value.to(device=device)

    return moved_inputs


def parse_action_output(last_action: Any) -> Dict[str, Any]:
    """
    中文：解析模型最后一次输出的 action。
    English: Parse the action output from the final inference run.
    """
    if last_action is None:
        # 中文：如果没有 action，返回空列表，并标记解析失败。
        # English: If no action is produced, return an empty list and mark parsing as failed.
        return {
            "parsed_action": [],
            "action_dimension": 0,
            "action_parse_success": False,
        }

    if hasattr(last_action, "tolist"):
        parsed_action = last_action.tolist()
    else:
        parsed_action = last_action

    if isinstance(parsed_action, (list, tuple)):
        action_dimension = len(parsed_action)
    else:
        action_dimension = 0

    return {
        "parsed_action": parsed_action,
        "action_dimension": action_dimension,
        "action_parse_success": True,
    }


def main() -> None:
    """中文：主函数。English: Main function."""
    parser = argparse.ArgumentParser(description="Run OpenVLA baseline inference.")

    parser.add_argument(
        "--model-id",
        type=str,
        default="openvla/openvla-7b",
        help="OpenVLA model id on Hugging Face.",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        default="",
        help="Path to the input image for baseline inference.",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the object",
        help="Language instruction for the robot task.",
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
        help="Number of warm-up inference runs.",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of measured inference runs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check script logic without loading OpenVLA.",
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/df_02_02_baseline_inference_log.md",
        help="Markdown log output path.",
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/tables/df_02_02_baseline_metrics.csv",
        help="CSV metrics output path.",
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)
    csv_path = Path(args.csv_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_commit = get_git_commit_hash()
    git_branch = get_git_branch()

    print_section("DF-02-02 OpenVLA Baseline Inference / OpenVLA Baseline 推理")

    torch_module = dynamic_import("torch")
    transformers_module = dynamic_import("transformers")

    log_lines = [
        "# DF-02-02 Baseline Inference Run",
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
        f"- Image path: `{args.image_path}`",
        f"- Instruction: `{args.instruction}`",
        f"- Warm-up runs: `{args.warmup_runs}`",
        f"- Measured runs: `{args.num_runs}`",
        f"- Dry run: `{args.dry_run}`",
        "",
    ]

    if torch_module is None:
        print("WARNING: PyTorch is not installed.")
        log_lines.append("## PyTorch")
        log_lines.append("- Status: Not installed")
        log_lines.append("")

    if transformers_module is None:
        print("WARNING: transformers is not installed.")
        log_lines.append("## Transformers")
        log_lines.append("- Status: Not installed")
        log_lines.append("")

    if args.dry_run:
        print("Dry run completed. No model was loaded.")

        log_lines.append("## Result")
        log_lines.append("- Dry run completed. No model was loaded.")
        log_lines.append("- This local run only checks script logic.")
        log_lines.append("")

        append_markdown_log(log_path, log_lines)
        return

    if torch_module is None or transformers_module is None:
        print("Baseline inference skipped because required dependencies are missing.")
        print("This is acceptable locally, but cloud GPU / HPC must install these dependencies.")

        log_lines.append("## Result")
        log_lines.append("- Baseline inference skipped because required dependencies are missing.")
        log_lines.append("- This is acceptable on the local machine.")
        log_lines.append("- Full baseline inference should be executed on cloud GPU / HPC.")
        log_lines.append("")

        append_markdown_log(log_path, log_lines)
        return

    if not args.image_path:
        raise ValueError("Please provide --image-path for real baseline inference.")

    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    memory_before_loading = get_cuda_memory_gb(torch_module)
    load_start = time.perf_counter()

    print_section("Loading Processor and Model / 加载 Processor 和模型")

    auto_processor = transformers_module.AutoProcessor
    auto_model_for_vision2seq = transformers_module.AutoModelForVision2Seq

    dtype = get_torch_dtype(torch_module, args.dtype)

    processor = auto_processor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    model = auto_model_for_vision2seq.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = model.to(args.device)
    model.eval()

    load_time = time.perf_counter() - load_start
    memory_after_loading = get_cuda_memory_gb(torch_module)

    print(f"Model loaded in {load_time:.2f} seconds.")

    image = load_image(args.image_path)

    # 中文：OpenVLA 常用 prompt 格式。
    # English: Common prompt format used by OpenVLA.
    prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"

    inputs = processor(
        prompt,
        image,
        return_tensors="pt",
    )

    inputs = move_inputs_to_device(
        inputs=dict(inputs),
        torch_module=torch_module,
        device=args.device,
        dtype=dtype,
    )

    print_section("Warm-up Inference / 预热推理")

    with torch_module.no_grad():
        for _ in range(args.warmup_runs):
            _ = model.predict_action(
                **inputs,
                unnorm_key=args.unnorm_key,
                do_sample=False,
            )

    print_section("Measured Inference / 正式计时推理")

    latency_list: List[float] = []
    last_action: Any = None

    with torch_module.no_grad():
        for run_index in range(args.num_runs):
            start_time = time.perf_counter()

            action = model.predict_action(
                **inputs,
                unnorm_key=args.unnorm_key,
                do_sample=False,
            )

            if torch_module.cuda.is_available():
                torch_module.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            latency_list.append(elapsed)
            last_action = action

            print(f"Run {run_index + 1}: {elapsed * 1000:.2f} ms")

    latency_summary = summarise_latency(latency_list)
    memory_after_inference = get_cuda_memory_gb(torch_module)
    action_info = parse_action_output(last_action)

    parsed_action = action_info["parsed_action"]
    action_dimension = action_info["action_dimension"]
    action_parse_success = action_info["action_parse_success"]

    result_row = {
        "timestamp": timestamp,
        "branch": git_branch,
        "commit": git_commit,
        "model_id": args.model_id,
        "device": args.device,
        "dtype": args.dtype,
        "warmup_runs": args.warmup_runs,
        "num_runs": args.num_runs,
        "load_time_seconds": round(load_time, 4),
        "avg_latency_ms": round(latency_summary["avg_ms"], 4),
        "median_latency_ms": round(latency_summary["median_ms"], 4),
        "min_latency_ms": round(latency_summary["min_ms"], 4),
        "max_latency_ms": round(latency_summary["max_ms"], 4),
        "std_latency_ms": round(latency_summary["std_ms"], 4),
        "vram_before_loading_gb": round(memory_before_loading["allocated_gb"], 4),
        "vram_after_loading_gb": round(memory_after_loading["allocated_gb"], 4),
        "vram_after_inference_gb": round(memory_after_inference["allocated_gb"], 4),
        "peak_vram_allocated_gb": round(memory_after_inference["max_allocated_gb"], 4),
        "peak_vram_reserved_gb": round(memory_after_inference["max_reserved_gb"], 4),
        "instruction": args.instruction,
        "image_path": args.image_path,
        "action_dimension": action_dimension,
        "action_parse_success": action_parse_success,
    }

    save_csv_result(csv_path, result_row)

    log_lines.extend(
        [
            "## Model Loading",
            f"- Model loading time: `{load_time:.4f} seconds`",
            f"- VRAM before loading: `{memory_before_loading['allocated_gb']:.4f} GB`",
            f"- VRAM after loading: `{memory_after_loading['allocated_gb']:.4f} GB`",
            "",
            "## Inference Latency",
            f"- Average latency: `{latency_summary['avg_ms']:.4f} ms`",
            f"- Median latency: `{latency_summary['median_ms']:.4f} ms`",
            f"- Minimum latency: `{latency_summary['min_ms']:.4f} ms`",
            f"- Maximum latency: `{latency_summary['max_ms']:.4f} ms`",
            f"- Standard deviation: `{latency_summary['std_ms']:.4f} ms`",
            "",
            "## VRAM Usage",
            f"- VRAM after inference: `{memory_after_inference['allocated_gb']:.4f} GB`",
            f"- Peak VRAM allocated: `{memory_after_inference['max_allocated_gb']:.4f} GB`",
            f"- Peak VRAM reserved: `{memory_after_inference['max_reserved_gb']:.4f} GB`",
            "",
            "## Sample Action Output",
            f"- Raw / parsed action: `{json.dumps(parsed_action, ensure_ascii=False, default=str)}`",
            f"- Action dimension: `{action_dimension}`",
            f"- Action parse success: `{action_parse_success}`",
            "",
            "## CSV Output",
            f"- Metrics CSV: `{csv_path}`",
            "",
        ]
    )

    append_markdown_log(log_path, log_lines)

    print_section("Baseline Finished / Baseline 完成")
    print(f"Average latency: {latency_summary['avg_ms']:.4f} ms")
    print(f"Peak VRAM allocated: {memory_after_inference['max_allocated_gb']:.4f} GB")
    print(f"Metrics saved to: {csv_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()