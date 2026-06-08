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

# 中文：下面导入的库大部分都是 Python 标准库，所以本地 dry-run 不需要先安装 OpenVLA 的重型依赖。
# English: Most imports below are Python standard-library modules, so local dry-run does not require heavy OpenVLA dependencies.
# 中文：argparse 用来读取命令行参数，例如 --dry-run、--device、--dtype。
# English: argparse reads command-line arguments such as --dry-run, --device, and --dtype.
import argparse
# 中文：csv 用来读取测试清单 CSV，也用来保存实验指标 CSV。
# English: csv reads the input instruction CSV and writes the output metrics CSV.
import csv
# 中文：importlib 用来动态导入 torch、transformers、PIL，避免本地没装这些库时脚本直接崩。
# English: importlib dynamically imports torch, transformers, and PIL so the script does not crash locally when they are missing.
import importlib
# 中文：json 用来把模型输出 action 转成字符串，方便保存到 CSV。
# English: json converts the model action output into a string that can be saved in CSV.
import json
# 中文：statistics 用来计算 latency 的平均值、中位数、标准差等统计量。
# English: statistics calculates latency mean, median, standard deviation, and related metrics.
import statistics
# 中文：subprocess 用来调用 git 命令，记录当前分支和 commit。
# English: subprocess runs git commands to record the current branch and commit.
import subprocess
# 中文：time.perf_counter() 用来做较准确的耗时统计。
# English: time.perf_counter() is used for more accurate timing measurement.
import time
# 中文：datetime 用来给每次实验记录时间戳。
# English: datetime adds a timestamp to each experiment run.
from datetime import datetime
# 中文：Path 让路径处理更稳定，Windows 和 Linux/HPC 都能用。
# English: Path makes file-path handling more stable across Windows and Linux/HPC.
from pathlib import Path
# 中文：typing 只用于类型标注，不改变运行逻辑。
# English: typing is only used for type hints and does not change runtime behavior.
from typing import Any, Dict, List, Optional


def print_section(title: str) -> None:
    """中文：打印分隔标题。English: Print a section title."""
    # 中文：先打印一个空行，再打印一条长分隔线，让终端输出更清楚。
    # English: Print a blank line and a long separator to make terminal output easier to read.
    print("\n" + "=" * 90)
    # 中文：打印当前阶段标题，例如加载模型、开始推理、测试完成。
    # English: Print the current stage title, such as model loading, inference, or finish.
    print(title)
    # 中文：再打印一条分隔线，形成一个完整的标题块。
    # English: Print another separator to complete the section block.
    print("=" * 90)


def dynamic_import(package_name: str) -> Optional[Any]:
    """
    中文：动态导入依赖，避免本地没有 torch / transformers / pillow 时直接报错。
    English: Dynamically import dependencies to avoid crashing when torch / transformers / pillow are missing locally.
    """
    # 中文：这里用 try 是为了让缺少某个可选依赖时返回 None，而不是立刻终止程序。
    # English: The try block returns None when an optional dependency is missing instead of stopping the script immediately.
    try:
        # 中文：如果依赖存在，就返回实际导入的模块对象。
        # English: If the dependency exists, return the imported module object.
        return importlib.import_module(package_name)
    # 中文：只有导入失败才进入这里，方便后面给出更友好的错误提示。
    # English: This branch is only used when import fails, so later code can show a clearer error message.
    except ImportError:
        return None


def run_command(command: List[str]) -> str:
    """
    中文：运行系统命令并返回输出。
    English: Run a system command and return its output.
    """
    try:
        # 中文：执行传入的命令，例如 git rev-parse 或 git branch。
        # English: Run the given command, for example git rev-parse or git branch.
        result = subprocess.run(
            # 中文：command 是字符串列表形式，避免手动拼接 shell 字符串导致转义问题。
            # English: command is a list of strings, avoiding shell-escaping issues from manual string concatenation.
            command,
            # 中文：capture_output=True 会同时捕获 stdout 和 stderr，不直接刷屏。
            # English: capture_output=True captures both stdout and stderr instead of printing them directly.
            capture_output=True,
            # 中文：text=True 让返回内容是普通字符串，而不是 bytes。
            # English: text=True returns normal strings instead of bytes.
            text=True,
            # 中文：check=False 表示命令失败也不抛异常，后面自己读取错误输出。
            # English: check=False means command failure will not raise an exception; stderr is handled manually below.
            check=False,
        )

        # 中文：优先返回正常输出，因为 git 命令成功时结果通常在 stdout 里。
        # English: Prefer stdout because successful git commands usually write their result there.
        if result.stdout:
            return result.stdout.strip()

        # 中文：如果没有正常输出但有错误输出，就把错误信息返回，方便写入日志排查。
        # English: If stdout is empty but stderr exists, return stderr for easier debugging in logs.
        if result.stderr:
            return result.stderr.strip()

        # 中文：如果命令没有任何输出，用 N/A 占位，避免后续日志出现空值。
        # English: If the command has no output, use N/A as a placeholder to avoid blank log values.
        return "N/A"

    # 中文：这里捕获所有异常，防止 git 不存在或当前目录不是 git 仓库时脚本中断。
    # English: Catch all exceptions so the script does not stop if git is unavailable or the folder is not a git repository.
    except Exception as error:
        return f"Command failed: {repr(error)}"


def get_git_commit_hash() -> str:
    """中文：获取当前 Git commit hash。English: Get current Git commit hash."""
    # 中文：记录短 commit hash，之后可以知道这次结果来自哪一次代码版本。
    # English: Record the short commit hash so later results can be linked to a specific code version.
    return run_command(["git", "rev-parse", "--short", "HEAD"])


def get_git_branch() -> str:
    """中文：获取当前 Git 分支。English: Get current Git branch."""
    # 中文：记录当前分支，方便区分 DF-02-03 等不同实验分支。
    # English: Record the current branch to distinguish experiment branches such as DF-02-03.
    return run_command(["git", "branch", "--show-current"])


def get_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """
    中文：根据字符串选择 torch dtype。
    English: Select torch dtype according to a string name.
    """
    # 中文：OpenVLA 云端测试默认使用 bfloat16，通常比 float32 更省显存。
    # English: OpenVLA cloud tests usually use bfloat16 by default because it saves VRAM compared with float32.
    if dtype_name == "bfloat16":
        return torch_module.bfloat16

    # 中文：float16 也是半精度，部分 GPU 上可能比 bfloat16 支持更好。
    # English: float16 is also half precision and may be better supported than bfloat16 on some GPUs.
    if dtype_name == "float16":
        return torch_module.float16

    # 中文：float32 精度更高，但显存占用明显更大，一般不适合 7B 模型 smoke test。
    # English: float32 has higher precision but much larger VRAM usage, so it is usually not ideal for a 7B smoke test.
    if dtype_name == "float32":
        return torch_module.float32

    # 中文：auto 交给 transformers / PyTorch 自己决定 dtype。
    # English: auto lets transformers / PyTorch decide the dtype automatically.
    return "auto"


def get_cuda_memory_gb(torch_module: Any) -> Dict[str, float]:
    """
    中文：获取 CUDA 显存使用情况。
    English: Get CUDA memory usage.
    """
    # 中文：如果 torch 没有导入成功，就无法读取 CUDA 信息，统一返回 0。
    # English: If torch was not imported successfully, CUDA information cannot be read, so return zeros.
    if torch_module is None:
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
            "total_vram_gb": 0.0,
        }

    # 中文：如果当前机器没有可用 CUDA，也返回 0；这适合本地 CPU dry-run 或无 GPU 环境。
    # English: If CUDA is not available, return zeros; this fits local CPU dry-run or non-GPU environments.
    if not torch_module.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
            "total_vram_gb": 0.0,
        }

    # 中文：读取当前 CUDA 设备编号，通常是 cuda:0。
    # English: Read the current CUDA device index, usually cuda:0.
    device_index = torch_module.cuda.current_device()
    # 中文：把 GPU 总显存从 bytes 转成 GB，方便日志阅读。
    # English: Convert total GPU memory from bytes to GB for easier log reading.
    total_vram_gb = torch_module.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)

    # 中文：allocated 是实际已分配显存，reserved 是 PyTorch 缓存池保留显存。
    # English: allocated is actively used memory, while reserved is memory kept by the PyTorch caching allocator.
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
    # 中文：没有 torch 或没有 CUDA 时无法读取 GPU 名字，所以返回 N/A。
    # English: Without torch or CUDA, the GPU name cannot be read, so return N/A.
    if torch_module is None or not torch_module.cuda.is_available():
        return "N/A"

    # 中文：只记录第 0 张 GPU 的名字，因为当前脚本默认单卡测试。
    # English: Record only GPU 0 because this script is designed as a single-GPU test.
    return torch_module.cuda.get_device_name(0)


def read_instruction_csv(csv_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    中文：读取包含 image_path 和 instruction 的 CSV。
    English: Read a CSV file containing image_path and instruction columns.
    """
    # 中文：先检查 CSV 是否存在；不存在就直接报错，避免后面出现更难懂的错误。
    # English: Check whether the CSV exists first; otherwise raise a clear error before later code fails.
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 中文：samples 用来保存每一条测试样本，包括图片路径、指令和备注。
    # English: samples stores each test sample, including image path, instruction, and note.
    samples: List[Dict[str, str]] = []

    # 中文：utf-8-sig 可以兼容带 BOM 的 CSV，Windows Excel 导出的文件更稳。
    # English: utf-8-sig supports CSV files with BOM, which is safer for files exported from Windows Excel.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        # 中文：DictReader 会把每一行读成字典，字段名来自 CSV 表头。
        # English: DictReader reads each row as a dictionary using the CSV header as field names.
        reader = csv.DictReader(file)

        # 中文：这个脚本最少需要 image_path 和 instruction 两列。
        # English: This script requires at least image_path and instruction columns.
        required_columns = {"image_path", "instruction"}
        # 中文：如果 CSV 没有表头，就无法知道哪一列是图片、哪一列是指令。
        # English: If the CSV has no header, the script cannot know which column is image path or instruction.
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")

        # 中文：检查必需列是否缺失，缺失时直接告诉用户缺哪些列。
        # English: Check missing required columns and report exactly which columns are missing.
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV file is missing columns: {missing_columns}")

        # 中文：逐行读取 CSV；row_index 用来生成稳定的 sample_id。
        # English: Read the CSV row by row; row_index is used to create a stable sample_id.
        for row_index, row in enumerate(reader):
            # 中文：读取图片路径，并去掉首尾空格，避免路径里有隐藏空格导致找不到文件。
            # English: Read the image path and strip spaces to avoid hidden-space path errors.
            image_path = str(row.get("image_path", "")).strip()
            # 中文：读取机器人任务指令，例如 pick up the object。
            # English: Read the robot task instruction, for example pick up the object.
            instruction = str(row.get("instruction", "")).strip()
            # 中文：note 是可选列，用来记录样本备注；没有也不会影响运行。
            # English: note is optional and stores sample remarks; missing notes do not affect execution.
            note = str(row.get("note", "")).strip()

            # 中文：如果图片路径或指令为空，就跳过这一行，避免无效样本进入推理。
            # English: If image path or instruction is empty, skip the row to avoid invalid inference samples.
            if not image_path or not instruction:
                continue

            # 中文：把有效样本加入列表，后面 dry-run 和真实推理都会使用这个列表。
            # English: Add the valid sample to the list used by both dry-run and real inference.
            samples.append(
                {
                    # 中文：sample_id 使用三位编号，方便日志和 CSV 对齐查看。
                    # English: sample_id uses a three-digit number for easier matching between logs and CSV.
                    "sample_id": f"sample_{row_index + 1:03d}",
                    "image_path": image_path,
                    "instruction": instruction,
                    "note": note,
                }
            )

            # 中文：如果设置了 limit，达到数量后就停止读取，避免一次跑太多样本。
            # English: If limit is set, stop reading once the limit is reached to avoid running too many samples.
            if limit is not None and len(samples) >= limit:
                break

    # 中文：返回清洗后的样本列表。
    # English: Return the cleaned sample list.
    return samples


def append_markdown_log(log_path: Path, lines: List[str]) -> None:
    """
    中文：追加写入 Markdown 日志。
    English: Append lines to a Markdown log file.
    """
    # 中文：如果日志目录不存在，就自动创建，例如 logs/。
    # English: Automatically create the log directory, for example logs/, if it does not exist.
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 中文：用追加模式 a，不覆盖旧日志；多次运行会连续写在同一个 Markdown 文件里。
    # English: Use append mode so old logs are not overwritten; multiple runs are added to the same Markdown file.
    with log_path.open("a", encoding="utf-8") as file:
        # 中文：把传入的多行文本合并成 Markdown 内容写入文件。
        # English: Join the input lines into Markdown content and write them to the file.
        file.write("\n".join(lines))
        # 中文：额外写两个换行，分开不同运行记录，日志更清楚。
        # English: Add two extra newlines to separate different runs in the log.
        file.write("\n\n")


def save_csv_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    中文：保存多行结果到 CSV。
    English: Save multiple result rows to a CSV file.
    """
    # 中文：如果没有结果行，就不创建空 CSV。
    # English: If there are no result rows, do not create an empty CSV.
    if not rows:
        return

    # 中文：确保输出目录存在，例如 results/tables/。
    # English: Ensure the output directory exists, for example results/tables/.
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 中文：用第一行的 key 作为 CSV 表头，保证列顺序稳定。
    # English: Use the first row keys as CSV headers to keep column order stable.
    fieldnames = list(rows[0].keys())

    # 中文：newline='' 可以避免 Windows 上 CSV 出现多余空行。
    # English: newline='' avoids extra blank lines in CSV files on Windows.
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        # 中文：DictWriter 按 fieldnames 写入字典格式的结果。
        # English: DictWriter writes dictionary-based rows according to fieldnames.
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # 中文：先写表头，再写所有实验结果行。
        # English: Write the header first, then all experiment result rows.
        writer.writeheader()
        writer.writerows(rows)


def summarise_latency_ms(values_ms: List[float]) -> Dict[str, float]:
    """
    中文：统计 latency 的平均值、中位数、最小值、最大值和标准差。
    English: Summarise latency using mean, median, min, max, and standard deviation.
    """
    # 中文：如果没有有效 latency，就全部返回 0，避免 statistics 对空列表报错。
    # English: If there are no valid latency values, return zeros to avoid statistics errors on an empty list.
    if not values_ms:
        return {
            "avg_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }

    # 中文：这里返回每个样本多次推理的统计结果。
    # English: Return summary statistics for repeated inference runs of one sample.
    return {
        "avg_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        # 中文：只有多于一个值时才计算标准差，否则标准差定义为 0。
        # English: Calculate standard deviation only when there is more than one value; otherwise use 0.
        "std_ms": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
    }


def load_image(image_path: Path) -> Any:
    """
    中文：加载输入图片。
    English: Load an input image.
    """
    # 中文：PIL 是 pillow 包的一部分，这里动态导入是为了本地 dry-run 不强制依赖 pillow。
    # English: PIL is part of pillow; dynamic import avoids requiring pillow during local dry-run.
    pil_image_module = dynamic_import("PIL.Image")

    # 中文：真实云端推理需要读取图片，所以没有 pillow 时必须明确报错。
    # English: Real cloud inference needs image loading, so the script must raise a clear error if pillow is missing.
    if pil_image_module is None:
        raise ImportError("PIL is not installed. Please install pillow on cloud GPU / HPC.")

    # 中文：统一转成 RGB，避免灰度图或带透明通道的图片影响 processor。
    # English: Convert to RGB so grayscale or transparent images do not confuse the processor.
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
    # 中文：创建一个新的字典保存移动后的输入，不直接改原始 inputs。
    # English: Create a new dictionary for moved inputs instead of modifying the original inputs directly.
    moved_inputs: Dict[str, Any] = {}

    # 中文：processor 输出可能包含 input_ids、pixel_values、attention_mask 等不同类型的数据。
    # English: Processor outputs may include input_ids, pixel_values, attention_mask, and other data types.
    for key, value in inputs.items():
        # 中文：如果某个值没有 .to() 方法，说明它不是 tensor，直接保留。
        # English: If a value has no .to() method, it is not a tensor, so keep it unchanged.
        if not hasattr(value, "to"):
            moved_inputs[key] = value
            continue

        # 中文：浮点 tensor 例如图像特征可以切换 dtype，以节省显存。
        # English: Floating tensors such as image features can change dtype to save VRAM.
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            # 中文：dtype 为 auto 时，只移动设备，不手动改精度。
            # English: When dtype is auto, move the tensor to the device without manually changing precision.
            if dtype == "auto":
                moved_inputs[key] = value.to(device=device)
            else:
                # 中文：指定 dtype 时，同时移动设备并转换浮点精度。
                # English: When dtype is specified, move the tensor and convert floating precision at the same time.
                moved_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            # 中文：整数 tensor 例如 token id 不能转成半精度，否则模型输入会错。
            # English: Integer tensors such as token ids must not be converted to half precision, otherwise model input will be wrong.
            moved_inputs[key] = value.to(device=device)

    # 中文：返回已经放到目标设备上的输入字典。
    # English: Return the input dictionary moved to the target device.
    return moved_inputs


def parse_action_output(action: Any) -> Dict[str, Any]:
    """
    中文：解析模型输出 action。
    English: Parse model action output.
    """
    # 中文：如果模型没有返回 action，说明这条样本推理失败。
    # English: If the model returns no action, this sample inference is treated as failed.
    if action is None:
        return {
            "parsed_action": [],
            "action_dimension": 0,
            "action_parse_success": False,
            "action_range_valid": False,
            "failure_reason": "no output",
        }

    # 中文：很多 PyTorch / NumPy 输出都有 tolist()，转成普通 Python list 更容易保存成 JSON。
    # English: Many PyTorch / NumPy outputs have tolist(); converting to a Python list makes JSON saving easier.
    if hasattr(action, "tolist"):
        parsed_action = action.tolist()
    else:
        parsed_action = action

    # 中文：如果 action 是 list 或 tuple，就用长度作为动作维度。
    # English: If action is a list or tuple, use its length as the action dimension.
    if isinstance(parsed_action, (list, tuple)):
        action_dimension = len(parsed_action)
    else:
        action_dimension = 0

    # 中文：这里只做非常轻量的检查，不假设具体机器人动作范围。
    # English: Only a lightweight check is used here because the exact robot action range is not assumed.
    action_range_valid = action_dimension > 0

    # 中文：把解析结果统一打包，方便后面直接写入 CSV。
    # English: Pack parsed results into one dictionary so they can be written to CSV later.
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
    # 中文：OpenVLA 常见输入格式是 In/Out 形式，这里把自然语言指令包进 prompt。
    # English: OpenVLA commonly uses an In/Out prompt format; the natural-language instruction is inserted here.
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
    # 中文：把样本里的相对/绝对图片路径转成 Path 对象，后面方便判断文件是否存在。
    # English: Convert the sample image path into a Path object for easier existence checks.
    image_path = Path(sample["image_path"])
    # 中文：取出当前样本的机器人指令，用来构造 OpenVLA prompt。
    # English: Extract the robot instruction of the current sample to build the OpenVLA prompt.
    instruction = sample["instruction"]

    # 中文：如果图片不存在，直接返回一行失败结果，而不是让整个 batch 中断。
    # English: If the image is missing, return one failed result row instead of stopping the whole batch.
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

    # 中文：end_to_end_start 记录从图片加载到 action 解析完成的总耗时起点。
    # English: end_to_end_start marks the start of total time from image loading to action parsing.
    end_to_end_start = time.perf_counter()

    # 中文：preprocess_start 只统计图片读取、prompt 构造和 processor 处理的时间。
    # English: preprocess_start measures only image loading, prompt building, and processor preparation time.
    preprocess_start = time.perf_counter()

    # 中文：加载当前样本图片，并统一转成 RGB。
    # English: Load the current sample image and convert it to RGB.
    image = load_image(image_path)
    # 中文：根据 instruction 生成模型输入 prompt。
    # English: Build the model input prompt from the instruction.
    prompt = make_prompt(instruction)

    # 中文：processor 把文字 prompt 和图片转换成模型可接受的 tensor 输入。
    # English: The processor converts the text prompt and image into tensor inputs accepted by the model.
    inputs = processor(
        prompt,
        image,
        return_tensors="pt",
    )

    # 中文：把 processor 输出移动到 GPU/CPU，并按 dtype 处理浮点 tensor。
    # English: Move processor outputs to GPU/CPU and apply dtype to floating tensors.
    inputs = move_inputs_to_device(
        inputs=dict(inputs),
        device=device,
        dtype=dtype,
    )

    # 中文：CUDA 是异步执行的，计时前同步一次可以让 preprocess 时间更准确。
    # English: CUDA is asynchronous, so synchronizing before timing makes preprocess time more accurate.
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    # 中文：把预处理耗时转换成毫秒，方便和 inference latency 对比。
    # English: Convert preprocessing time to milliseconds for comparison with inference latency.
    preprocess_time_ms = (time.perf_counter() - preprocess_start) * 1000

    # 中文：no_grad 关闭梯度计算，推理时更省显存也更快。
    # English: no_grad disables gradient calculation, saving VRAM and improving inference speed.
    with torch_module.no_grad():
        # 中文：warmup_runs 不计入最终 latency，主要用于让 GPU 和模型进入稳定状态。
        # English: warmup_runs are not counted in final latency; they help GPU/model reach a stable state.
        for _ in range(warmup_runs):
            # 中文：predict_action 是 OpenVLA 的动作预测接口，输出机器人动作向量。
            # English: predict_action is the OpenVLA action prediction API and outputs a robot action vector.
            _ = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

    # 中文：latency_values_ms 保存每次正式测量的推理耗时。
    # English: latency_values_ms stores inference time for each measured run.
    latency_values_ms: List[float] = []
    # 中文：last_action 保存最后一次输出，用于后面解析动作维度和 JSON。
    # English: last_action keeps the final output for later action dimension and JSON parsing.
    last_action: Any = None

    # 中文：正式测量阶段同样关闭梯度计算。
    # English: The measured inference stage also runs with gradient calculation disabled.
    with torch_module.no_grad():
        # 中文：每个样本可以测多次，然后统计平均 latency。
        # English: Each sample can be measured multiple times, then averaged for latency statistics.
        for _ in range(num_runs):
            # 中文：开始计时前同步 GPU，避免把之前未完成的 CUDA 操作算进本次结果。
            # English: Synchronize before timing to avoid including unfinished earlier CUDA operations in this run.
            if torch_module.cuda.is_available():
                torch_module.cuda.synchronize()

            # 中文：记录单次推理开始时间。
            # English: Record the start time of one inference run.
            inference_start = time.perf_counter()

            # 中文：执行一次真实动作预测。
            # English: Run one real action prediction.
            action = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )

            # 中文：推理结束后再次同步，确保 GPU 计算真的完成再停止计时。
            # English: Synchronize after inference so timing stops only after GPU computation has actually finished.
            if torch_module.cuda.is_available():
                torch_module.cuda.synchronize()

            # 中文：计算本次推理耗时，并转换成毫秒。
            # English: Calculate this inference duration and convert it to milliseconds.
            inference_time_ms = (time.perf_counter() - inference_start) * 1000
            # 中文：保存本次 latency，用于后面计算 mean/median/min/max/std。
            # English: Save this latency for later mean/median/min/max/std calculation.
            latency_values_ms.append(inference_time_ms)
            last_action = action

    # 中文：计算端到端耗时，包括预处理、warmup、正式推理和 action 解析前的总时间。
    # English: Calculate end-to-end time, including preprocessing, warmup, measured inference, and time before action parsing.
    end_to_end_time_ms = (time.perf_counter() - end_to_end_start) * 1000

    # 中文：对当前样本的多次推理 latency 做统计汇总。
    # English: Summarise repeated inference latencies for the current sample.
    latency_summary = summarise_latency_ms(latency_values_ms)
    # 中文：解析最后一次模型输出，得到 action 维度、是否成功等信息。
    # English: Parse the last model output to get action dimension, success flag, and related information.
    action_info = parse_action_output(last_action)

    # 中文：返回一行结构化结果，后面会合并 GPU/commit 信息后写入 CSV。
    # English: Return one structured result row that will later be merged with GPU/commit information and written to CSV.
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
    # 中文：创建命令行参数解析器，用户可以在终端传入不同配置。
    # English: Create the command-line parser so users can pass different configurations from the terminal.
    parser = argparse.ArgumentParser(description="Run OpenVLA baseline batch smoke test.")

    # 中文：输入 CSV 参数：告诉脚本从哪个表读取测试图片和指令。
    # English: Input CSV argument: tells the script which table contains test images and instructions.
    parser.add_argument(
        "--instruction-csv",
        type=str,
        default="test_instructions.csv",
        help="CSV file with image_path and instruction columns.",
    )

    # 中文：模型 ID 参数：默认使用 Hugging Face 上的 openvla/openvla-7b。
    # English: Model ID argument: by default uses openvla/openvla-7b on Hugging Face.
    parser.add_argument(
        "--model-id",
        type=str,
        default="openvla/openvla-7b",
        help="OpenVLA model id on Hugging Face.",
    )

    # 中文：unnorm-key 参数：OpenVLA 用它把归一化动作还原到对应数据集的动作空间。
    # English: unnorm-key argument: OpenVLA uses it to unnormalize actions into the target dataset action space.
    parser.add_argument(
        "--unnorm-key",
        type=str,
        default="bridge_orig",
        help="Action unnormalization key used by OpenVLA.",
    )

    # 中文：device 参数：云端一般用 cuda:0，本地 dry-run 不会真正加载模型。
    # English: device argument: cloud runs usually use cuda:0; local dry-run does not load the model.
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference, for example cuda:0 or cpu.",
    )

    # 中文：dtype 参数：控制模型精度，影响显存占用和推理速度。
    # English: dtype argument: controls model precision, affecting VRAM usage and inference speed.
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype.",
    )

    # 中文：warmup-runs 参数：每个样本正式计时前先跑几次预热。
    # English: warmup-runs argument: number of warm-up runs before measured inference for each sample.
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warm-up inference runs per sample.",
    )

    # 中文：num-runs 参数：每个样本正式计时的重复次数。
    # English: num-runs argument: number of measured inference repeats for each sample.
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of measured inference runs per sample.",
    )

    # 中文：limit 参数：限制最多读取多少个样本，smoke test 阶段默认 10 张图够用。
    # English: limit argument: limits maximum samples; 10 images are enough for the smoke-test stage by default.
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of samples to run.",
    )

    # 中文：dry-run 参数：只检查 CSV 和路径，不加载模型，适合本地 Windows 快速检查。
    # English: dry-run argument: checks only CSV and paths without loading the model, suitable for quick local Windows checks.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check CSV and paths without loading OpenVLA.",
    )

    # 中文：log-path 参数：Markdown 日志输出位置。
    # English: log-path argument: output path for the Markdown log.
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/df_02_03_cloud_smoke_test_log.md",
        help="Markdown log output path.",
    )

    # 中文：csv-path 参数：实验指标 CSV 输出位置。
    # English: csv-path argument: output path for the metrics CSV.
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/tables/df_02_03_cloud_smoke_test_metrics.csv",
        help="CSV metrics output path.",
    )

    # 中文：解析终端传入的参数，后面统一通过 args 使用。
    # English: Parse terminal arguments and access them through args later.
    args = parser.parse_args()

    # 中文：把字符串路径转成 Path 对象，方便跨平台处理。
    # English: Convert string paths into Path objects for cross-platform handling.
    instruction_csv = Path(args.instruction_csv)
    log_path = Path(args.log_path)
    csv_path = Path(args.csv_path)

    # 中文：记录当前运行时间，用于日志和 CSV 追踪。
    # English: Record the current run time for tracking in logs and CSV.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 中文：记录当前 Git 分支，方便知道结果来自哪个任务分支。
    # English: Record the current Git branch to know which task branch produced the result.
    git_branch = get_git_branch()
    # 中文：记录当前 Git commit，方便实验复现和论文记录。
    # English: Record the current Git commit for reproducibility and dissertation notes.
    git_commit = get_git_commit_hash()

    # 中文：打印脚本启动标题。
    # English: Print the script start title.
    print_section("DF-02-03 OpenVLA Batch Baseline Smoke Test / 批量 Baseline 跑通测试")

    # 中文：读取测试样本；dry-run 和真实推理都会先做这一步。
    # English: Read test samples; both dry-run and real inference do this first.
    samples = read_instruction_csv(instruction_csv, limit=args.limit)

    # 中文：在终端打印 CSV 路径和样本数量，方便确认读到的是正确文件。
    # English: Print CSV path and sample count in the terminal to confirm the correct file was read.
    print(f"CSV path: {instruction_csv}")
    print(f"Loaded samples: {len(samples)}")

    # 中文：准备 Markdown 日志内容，先记录时间、Git 信息和运行配置。
    # English: Prepare Markdown log content, starting with timestamp, Git information, and run configuration.
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

    # 中文：逐个样本检查图片是否存在，并把样本列表写入日志。
    # English: Check whether each sample image exists and write the sample list to the log.
    for sample in samples:
        # 中文：这里只检查路径可见性，不会打开图片。
        # English: This only checks path visibility and does not open the image.
        image_exists = Path(sample["image_path"]).exists()
        print(f"{sample['sample_id']}: {sample['image_path']} | exists={image_exists} | {sample['instruction']}")
        # 中文：把每个样本的信息追加到 Markdown 日志，方便云端运行后回看。
        # English: Append each sample's information to the Markdown log for later review after cloud runs.
        log_lines.append(
            f"- `{sample['sample_id']}` | `{sample['image_path']}` | exists=`{image_exists}` | instruction=`{sample['instruction']}`"
        )

    log_lines.append("")

    # 中文：dry-run 分支到这里就结束，不会导入 torch/transformers，也不会加载 OpenVLA。
    # English: The dry-run branch ends here; it does not import torch/transformers or load OpenVLA.
    if args.dry_run:
        # 中文：打印 dry-run 完成提示。
        # English: Print the dry-run completion message.
        print_section("Dry Run Finished / Dry Run 完成")
        print("No model was loaded.")
        print("This only checks whether the CSV can be read and image paths are visible.")

        # 中文：把 dry-run 的结果写进 Markdown 日志。
        # English: Write the dry-run result into the Markdown log.
        log_lines.extend(
            [
                "## Dry Run Result",
                "- No model was loaded.",
                "- CSV reading succeeded.",
                "- Image path visibility was checked.",
                "",
            ]
        )

        # 中文：保存 dry-run 日志后直接返回，避免继续执行真实模型加载逻辑。
        # English: Save the dry-run log and return immediately to avoid real model-loading logic.
        append_markdown_log(log_path, log_lines)
        return

    # 中文：真实推理才会走到这里，开始动态导入 PyTorch。
    # English: Real inference reaches this point and starts dynamically importing PyTorch.
    torch_module = dynamic_import("torch")
    # 中文：动态导入 transformers，用于加载 OpenVLA processor 和 model。
    # English: Dynamically import transformers to load the OpenVLA processor and model.
    transformers_module = dynamic_import("transformers")

    # 中文：如果缺少 torch 或 transformers，说明当前环境还不能跑真实 OpenVLA 推理。
    # English: If torch or transformers is missing, the current environment cannot run real OpenVLA inference.
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

    # 中文：如果 CUDA 可用，先清空峰值显存统计，保证本次记录更干净。
    # English: If CUDA is available, reset peak memory stats so this run's VRAM record is cleaner.
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    # 中文：把命令行 dtype 字符串转换成 torch dtype 对象。
    # English: Convert the command-line dtype string into a torch dtype object.
    dtype = get_torch_dtype(torch_module, args.dtype)

    # 中文：记录模型加载前的显存，用来计算加载 OpenVLA 后显存变化。
    # English: Record VRAM before model loading to compare against VRAM after loading OpenVLA.
    memory_before_loading = get_cuda_memory_gb(torch_module)
    # 中文：记录 GPU 名称，方便不同租用 GPU 之间做对比。
    # English: Record the GPU name for comparison across rented GPU types.
    gpu_name = get_gpu_name(torch_module)

    # 中文：进入模型加载阶段；这里设计为整个 batch 只加载一次模型。
    # English: Enter model-loading stage; the model is loaded only once for the whole batch.
    print_section("Loading OpenVLA Once / 只加载一次 OpenVLA")

    # 中文：开始统计模型加载时间。
    # English: Start measuring model loading time.
    load_start = time.perf_counter()

    # 中文：加载 OpenVLA 对应的 processor，负责文字和图片预处理。
    # English: Load the processor for OpenVLA, which handles text and image preprocessing.
    processor = transformers_module.AutoProcessor.from_pretrained(
        # 中文：使用命令行传入的模型 ID，默认是 openvla/openvla-7b。
        # English: Use the model ID from command-line arguments; default is openvla/openvla-7b.
        args.model_id,
        # 中文：OpenVLA 需要 trust_remote_code=True 才能加载自定义模型代码。
        # English: OpenVLA needs trust_remote_code=True to load its custom model code.
        trust_remote_code=True,
    )

    # 中文：加载 Vision2Seq 模型本体，这是最占显存和最耗时的一步。
    # English: Load the Vision2Seq model itself; this is the most VRAM-heavy and time-consuming step.
    model = transformers_module.AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        # 中文：使用前面解析出的 dtype，例如 bfloat16。
        # English: Use the dtype parsed above, for example bfloat16.
        torch_dtype=dtype,
        # 中文：low_cpu_mem_usage=True 可以降低加载过程中 CPU 内存压力。
        # English: low_cpu_mem_usage=True can reduce CPU memory pressure during loading.
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 中文：把模型移动到目标设备，例如 cuda:0。
    # English: Move the model to the target device, such as cuda:0.
    model = model.to(args.device)
    # 中文：切换到 eval 模式，关闭 dropout 等训练行为。
    # English: Switch to eval mode to disable training behaviours such as dropout.
    model.eval()

    # 中文：模型移动到 GPU 后同步一次，保证加载计时准确结束。
    # English: Synchronize after moving the model to GPU so loading time ends accurately.
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    # 中文：计算模型加载总耗时。
    # English: Calculate total model loading time.
    model_load_time_seconds = time.perf_counter() - load_start
    # 中文：记录模型加载完成后的显存占用。
    # English: Record VRAM usage after model loading.
    memory_after_loading = get_cuda_memory_gb(torch_module)

    # 中文：把关键加载信息打印到终端，方便云端实时查看。
    # English: Print key loading information to the terminal for real-time cloud checking.
    print(f"GPU: {gpu_name}")
    print(f"Model loaded in {model_load_time_seconds:.2f} seconds.")
    print(f"VRAM after loading: {memory_after_loading['allocated_gb']:.4f} GB")

    # 中文：result_rows 保存所有样本的结果，最后统一写入 CSV。
    # English: result_rows stores all sample results and is written to CSV at the end.
    result_rows: List[Dict[str, Any]] = []

    # 中文：进入批量推理阶段。
    # English: Enter the batch inference stage.
    print_section("Running Batch Inference / 开始批量推理")

    # 中文：逐个样本运行推理；模型已经加载好，不会每张图重新加载一次。
    # English: Run inference sample by sample; the model is already loaded and will not reload for every image.
    for sample_index, sample in enumerate(samples):
        # 中文：打印当前进度，例如 Running 1/10。
        # English: Print current progress, for example Running 1/10.
        print(f"\nRunning {sample_index + 1}/{len(samples)}: {sample['sample_id']}")

        # 中文：对单个样本执行完整流程：图片加载、processor、warmup、推理和 action 解析。
        # English: Run the full pipeline for one sample: image loading, processor, warmup, inference, and action parsing.
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

        # 中文：每个样本结束后记录一次显存，观察是否有显存持续增长的问题。
        # English: Record VRAM after each sample to check whether memory usage keeps growing.
        memory_after_sample = get_cuda_memory_gb(torch_module)

        # 中文：把全局实验信息和当前样本结果合并成一行 CSV 数据。
        # English: Merge global experiment information and the current sample result into one CSV row.
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
            # 中文：记录模型加载前显存。
            # English: Record VRAM before model loading.
            "vram_before_loading_gb": round(memory_before_loading["allocated_gb"], 4),
            # 中文：记录模型加载后显存。
            # English: Record VRAM after model loading.
            "vram_after_loading_gb": round(memory_after_loading["allocated_gb"], 4),
            # 中文：记录当前样本结束后的显存。
            # English: Record VRAM after the current sample.
            "vram_after_sample_gb": round(memory_after_sample["allocated_gb"], 4),
            # 中文：记录到目前为止的峰值已分配显存。
            # English: Record peak allocated VRAM so far.
            "peak_vram_allocated_gb": round(memory_after_sample["max_allocated_gb"], 4),
            # 中文：记录到目前为止的峰值 reserved 显存。
            # English: Record peak reserved VRAM so far.
            "peak_vram_reserved_gb": round(memory_after_sample["max_reserved_gb"], 4),
            "gpu_total_vram_gb": round(memory_after_sample["total_vram_gb"], 4),
            # 中文：把 run_single_sample 返回的样本指标展开到同一行。
            # English: Expand the metrics returned by run_single_sample into the same row.
            **sample_result,
        }

        # 中文：把当前样本结果加入总结果列表。
        # English: Add the current sample result to the full result list.
        result_rows.append(row)

        # 中文：在终端打印当前样本的核心结果，方便边跑边看。
        # English: Print key results of the current sample for live monitoring.
        print(f"Average latency: {row['avg_latency_ms']} ms")
        print(f"Action parse success: {row['action_parse_success']}")
        print(f"Peak VRAM allocated: {row['peak_vram_allocated_gb']} GB")

    # 中文：所有样本跑完后，把结果保存成 CSV。
    # English: After all samples finish, save results to CSV.
    save_csv_rows(csv_path, result_rows)

    # 中文：只把 action 解析成功的样本 latency 用于整体 latency 统计。
    # English: Use only action-parse-success samples for overall latency statistics.
    all_avg_latencies = [
        float(row["avg_latency_ms"])
        for row in result_rows
        if row.get("action_parse_success") is True
    ]
    overall_latency = summarise_latency_ms(all_avg_latencies)

    # 中文：统计成功解析 action 的样本数量。
    # English: Count how many samples successfully produced parsable actions.
    success_count = sum(1 for row in result_rows if row.get("action_parse_success") is True)
    # 中文：统计总样本数量。
    # English: Count the total number of result rows.
    total_count = len(result_rows)
    # 中文：计算 action 解析成功率；如果没有样本则返回 0，避免除以 0。
    # English: Calculate action parse success rate; return 0 if there are no samples to avoid division by zero.
    parse_success_rate = success_count / total_count if total_count > 0 else 0.0

    # 中文：读取最终显存状态，包括峰值显存。
    # English: Read final VRAM state, including peak memory.
    final_memory = get_cuda_memory_gb(torch_module)

    # 中文：把云端真实运行结果追加到 Markdown 日志。
    # English: Append real cloud-run results to the Markdown log.
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

    # 中文：保存完整 Markdown 日志。
    # English: Save the complete Markdown log.
    append_markdown_log(log_path, log_lines)

    # 中文：终端输出最终摘要，方便快速确认本次 smoke test 是否成功。
    # English: Print the final summary in the terminal to quickly check whether the smoke test succeeded.
    print_section("Batch Smoke Test Finished / 批量跑通测试完成")
    print(f"Total samples: {total_count}")
    print(f"Action parse success rate: {parse_success_rate:.4f}")
    print(f"Overall avg latency: {overall_latency['avg_ms']:.4f} ms")
    print(f"Peak VRAM allocated: {final_memory['max_allocated_gb']:.4f} GB")
    print(f"CSV saved to: {csv_path}")
    print(f"Log saved to: {log_path}")


# 中文：只有直接运行这个脚本时才执行 main；如果被其他文件 import，则不会自动运行。
# English: Run main only when this script is executed directly; importing this file will not trigger execution.
if __name__ == "__main__":
    main()