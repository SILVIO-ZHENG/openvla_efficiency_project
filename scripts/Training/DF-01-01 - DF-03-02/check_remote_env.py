"""
中文：检查远端 GPU / HPC 环境是否适合运行 OpenVLA 项目。
English: Check whether the remote GPU / HPC environment is suitable for the OpenVLA project.
"""

import sys
import platform
import subprocess
import importlib

def print_section(title: str) -> None:
    """中文：打印分隔标题。
    English: Print a section title."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_python_version() -> None:
    """中文：检查 Python 版本是否为 3.10。
    English: Check whether Python version is 3.10."""
    print_section("Python Version Check / Python 版本检查")

    version = sys.version
    major = sys.version_info.major
    minor = sys.version_info.minor

    print(f"Python version: {version}")

    if major == 3 and minor == 10:
        print("Status: OK - Python 3.10 is being used.")
    else:
        print("Status: WARNING - This project should use Python 3.10.")


def check_system_info() -> None:
    """中文：检查操作系统信息。English: Check operating system information."""
    print_section("System Information / 系统信息")

    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()}")
    print(f"Machine: {platform.machine()}")


def check_nvidia_smi() -> None:
    """中文：检查 nvidia-smi 是否可用。
    English: Check whether nvidia-smi is available."""
    print_section("NVIDIA-SMI Check / NVIDIA-SMI 检查")

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("Status: OK - nvidia-smi is available.")
            print(result.stdout)
        else:
            print("Status: WARNING - nvidia-smi returned an error.")
            print(result.stderr)
    except FileNotFoundError:
        print("Status: WARNING - nvidia-smi is not available on this machine.")


def check_torch_cuda() -> None:
    """中文：检查 PyTorch 和 CUDA 状态。
    English: Check PyTorch and CUDA status."""
    print_section("PyTorch and CUDA Check / PyTorch 和 CUDA 检查")

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        print("Status: WARNING - PyTorch is not installed.")
        print("This is acceptable on the local machine, but PyTorch is required on cloud GPU / HPC.")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for gpu_id in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(gpu_id)
            total_memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
            print(f"GPU {gpu_id}: {gpu_name}")
            print(f"GPU {gpu_id} total memory: {total_memory_gb:.2f} GB")
    else:
        print("Status: WARNING - CUDA is not available.")
        print("This is normal on a local laptop without GPU CUDA setup.")
        print("For real OpenVLA loading, this should be OK on the cloud GPU / HPC machine.")


def main() -> None:
    """中文：运行所有环境检查。
    English: Run all environment checks."""
    print_section("OpenVLA Remote Environment Check / OpenVLA 远端环境检查")

    check_python_version()
    check_system_info()
    check_nvidia_smi()
    check_torch_cuda()

    print_section("Check Finished / 检查完成")
    print("This script is used before loading OpenVLA on cloud GPU / HPC.")


if __name__ == "__main__":
    main()