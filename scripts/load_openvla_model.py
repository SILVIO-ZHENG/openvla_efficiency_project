"""
中文：在云端 GPU / HPC 上加载 OpenVLA 模型，并记录基础加载信息。
English: Load the OpenVLA model on cloud GPU / HPC and record basic loading information.

说明 / Note:
- 本地电脑主要用于代码编辑和轻量检查。
- 真正的 OpenVLA 模型加载应该在云端 GPU / 学校 HPC 上运行。
- 如果本地没有 torch / transformers，这是正常的。
"""

import argparse
import importlib
import time
from pathlib import Path
from datetime import datetime


def print_section(title: str) -> None:
    """中文：打印分隔标题。English: Print a section title."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def dynamic_import(package_name: str):
    """
    中文：动态导入依赖包，避免本地没安装重型依赖时直接报错。
    English: Dynamically import packages to avoid crashing when heavy dependencies are not installed locally.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


def write_log(log_path: Path, content: str) -> None:
    """中文：把运行结果写入日志文件。English: Write running result into a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(content + "\n")


def get_torch_dtype(torch_module, dtype_name: str):
    """
    中文：根据字符串选择 torch dtype。
    English: Select torch dtype according to the given string.
    """
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    return "auto"


def main() -> None:
    """中文：主函数。English: Main function."""
    parser = argparse.ArgumentParser(description="Load OpenVLA model on cloud GPU / HPC.")

    parser.add_argument(
        "--model-id",
        type=str,
        default="openvla/openvla-7b",
        help="Hugging Face model id for OpenVLA.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model loading, for example: auto, cuda:0, cpu.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model loading dtype.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check dependencies and configuration without loading the model.",
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/df_02_01_first_model_loading_log.md",
        help="Path to save model loading log.",
    )

    args = parser.parse_args()
    log_path = Path(args.log_path)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print_section("OpenVLA Model Loading Script / OpenVLA 模型加载脚本")

    log_lines = [
        "# DF-02-01 First OpenVLA Model Loading Log",
        "",
        f"## Timestamp",
        f"- {timestamp}",
        "",
        "## Configuration",
        f"- Model ID: `{args.model_id}`",
        f"- Device: `{args.device}`",
        f"- Dtype: `{args.dtype}`",
        f"- Dry run: `{args.dry_run}`",
        "",
    ]

    torch = dynamic_import("torch")
    transformers = dynamic_import("transformers")

    if torch is None:
        print("WARNING: PyTorch is not installed.")
        print("This is acceptable on the local machine, but required on cloud GPU / HPC.")
        log_lines.append("## PyTorch")
        log_lines.append("- Status: Not installed")
        log_lines.append("")
    else:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        log_lines.append("## PyTorch")
        log_lines.append(f"- Version: `{torch.__version__}`")
        log_lines.append(f"- CUDA available: `{torch.cuda.is_available()}`")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            print(f"GPU 0: {gpu_name}")
            print(f"GPU 0 total memory: {total_memory_gb:.2f} GB")

            log_lines.append(f"- GPU 0: `{gpu_name}`")
            log_lines.append(f"- GPU 0 memory: `{total_memory_gb:.2f} GB`")

        log_lines.append("")

    if transformers is None:
        print("WARNING: transformers is not installed.")
        print("This is acceptable locally, but required before loading OpenVLA on cloud GPU / HPC.")
        log_lines.append("## Transformers")
        log_lines.append("- Status: Not installed")
        log_lines.append("")
    else:
        print(f"Transformers version: {transformers.__version__}")
        log_lines.append("## Transformers")
        log_lines.append(f"- Version: `{transformers.__version__}`")
        log_lines.append("")

    if args.dry_run:
        print_section("Dry Run Finished / Dry Run 完成")
        print("No model was loaded.")
        log_lines.append("## Result")
        log_lines.append("- Dry run completed. No model was loaded.")
        log_lines.append("")
        write_log(log_path, "\n".join(log_lines))
        return

    if torch is None or transformers is None:
        print_section("Model Loading Skipped / 跳过模型加载")
        print("PyTorch and transformers are required for real OpenVLA loading.")
        print("Install them on cloud GPU / HPC before running this script without --dry-run.")

        log_lines.append("## Result")
        log_lines.append("- Model loading skipped because required dependencies are missing.")
        log_lines.append("")
        write_log(log_path, "\n".join(log_lines))
        return

    print_section("Loading OpenVLA / 正在加载 OpenVLA")
    print(f"Loading model: {args.model_id}")

    AutoProcessor = transformers.AutoProcessor
    AutoModelForVision2Seq = transformers.AutoModelForVision2Seq

    dtype = get_torch_dtype(torch, args.dtype)

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )

        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        if args.device != "auto":
            model = model.to(args.device)

        model.eval()

        elapsed_time = time.time() - start_time

        print("Status: OK - OpenVLA model loaded successfully.")
        print(f"Loading time: {elapsed_time:.2f} seconds")

        log_lines.append("## Result")
        log_lines.append("- Status: OpenVLA model loaded successfully")
        log_lines.append(f"- Loading time: `{elapsed_time:.2f} seconds`")
        log_lines.append("")

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

            print(f"CUDA memory allocated: {allocated_gb:.2f} GB")
            print(f"CUDA memory reserved: {reserved_gb:.2f} GB")

            log_lines.append("## CUDA Memory")
            log_lines.append(f"- Allocated: `{allocated_gb:.2f} GB`")
            log_lines.append(f"- Reserved: `{reserved_gb:.2f} GB`")
            log_lines.append("")

        write_log(log_path, "\n".join(log_lines))

        # 中文：避免变量被误删前没有使用的提示
        # English: Keep references clear for debugging.
        _ = processor
        _ = model

    except Exception as error:
        elapsed_time = time.time() - start_time

        print("Status: ERROR - OpenVLA model loading failed.")
        print(f"Error: {error}")

        log_lines.append("## Result")
        log_lines.append("- Status: Failed")
        log_lines.append(f"- Loading time before failure: `{elapsed_time:.2f} seconds`")
        log_lines.append(f"- Error: `{repr(error)}`")
        log_lines.append("")

        write_log(log_path, "\n".join(log_lines))


if __name__ == "__main__":
    main()