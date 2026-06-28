# Notes:
# These cloud paths are local paths on the author's cloud server.
# They are used only for experiment reproduction and do not contain access credentials.
#
# DF-04-01 LoRA training wrapper.
# This script does not implement a new trainer.
# It connects this project to the official OpenVLA finetune.py script,
# sets the fixed LIBERO split JSON, and records required experiment fields.

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# This function loads the YAML config file for the LoRA training run.
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# This function checks whether the required config blocks and keys exist.
def validate_config(config: dict) -> None:
    required_blocks = ["run", "paths", "model", "dataset", "lora", "training", "logging"]

    for block in required_blocks:
        if block not in config:
            raise KeyError(f"Missing required config block: {block}")

    required_path_keys = [
        "project_root",
        "official_openvla_root",
        "official_finetune_py",
        "dataset_root",
        "split_json",
        "output_dir",
        "log_path",
        "summary_csv",
        "metadata_json",
    ]

    for key in required_path_keys:
        if key not in config["paths"]:
            raise KeyError(f"Missing required paths.{key}")

    required_training_keys = [
        "batch_size",
        "max_steps",
        "save_steps",
        "learning_rate",
        "grad_accumulation_steps",
        "image_aug",
        "shuffle_buffer_size",
        "save_latest_checkpoint_only",
        "expected_train_episodes",
    ]

    for key in required_training_keys:
        if key not in config["training"]:
            raise KeyError(f"Missing required training.{key}")


# This function sets the random seed for Python, NumPy, and PyTorch.
def set_random_seed(seed: int) -> None:
    random.seed(seed)

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


# This function checks whether important Python packages are installed.
def check_dependencies() -> dict:
    dependency_status = {}

    for package_name in ["torch", "transformers", "peft", "accelerate", "yaml", "tensorflow", "tensorflow_datasets"]:
        try:
            __import__(package_name)
            dependency_status[package_name] = "installed"
        except ImportError:
            dependency_status[package_name] = "not_installed"

    return dependency_status


# This function collects GPU and CUDA information before training starts.
def get_gpu_info() -> dict:
    gpu_info = {
        "cuda_available": False,
        "gpu_name": "not_available",
        "gpu_count": 0,
        "torch_version": "not_available",
        "cuda_version": "not_available",
        "peak_vram_allocated_gb": None,
        "peak_vram_reserved_gb": None,
    }

    try:
        import torch

        gpu_info["torch_version"] = torch.__version__
        gpu_info["cuda_version"] = str(torch.version.cuda)
        gpu_info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
            gpu_info["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 4)
    except ImportError:
        pass

    return gpu_info


# This function creates required output folders before training starts.
def prepare_output_dirs(config: dict) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    log_path = Path(config["paths"]["log_path"])
    summary_csv = Path(config["paths"]["summary_csv"])
    metadata_json = Path(config["paths"]["metadata_json"])

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)


# This function checks that the official OpenVLA split patch is present.
def check_openvla_split_patch(config: dict) -> None:
    dataset_py = Path(config["paths"]["official_openvla_root"]) / "prismatic/vla/datasets/rlds/dataset.py"
    mixtures_py = Path(config["paths"]["official_openvla_root"]) / "prismatic/vla/datasets/rlds/oxe/mixtures.py"

    if not dataset_py.exists():
        raise FileNotFoundError(f"Official OpenVLA dataset.py was not found: {dataset_py}")

    if not mixtures_py.exists():
        raise FileNotFoundError(f"Official OpenVLA mixtures.py was not found: {mixtures_py}")

    dataset_text = dataset_py.read_text(encoding="utf-8")
    mixtures_text = mixtures_py.read_text(encoding="utf-8")

    if "OPENVLA_EPISODE_SPLIT_JSON" not in dataset_text:
        raise RuntimeError("OPENVLA_EPISODE_SPLIT_JSON patch was not found in official dataset.py.")

    if config["dataset"]["mixture_name"] not in mixtures_text:
        raise RuntimeError(f"{config['dataset']['mixture_name']} was not found in official mixtures.py.")


# This function checks all important files and paths before full training starts.
def run_preflight_checks(config: dict) -> dict:
    dataset_root = Path(config["paths"]["dataset_root"])
    split_json = Path(config["paths"]["split_json"])
    official_finetune = Path(config["paths"]["official_finetune_py"])

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if not split_json.exists():
        raise FileNotFoundError(f"Split JSON does not exist: {split_json}")

    if not official_finetune.exists():
        raise FileNotFoundError(f"Official OpenVLA finetune.py does not exist: {official_finetune}")

    with split_json.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    if not split_data:
        raise ValueError("Split JSON is empty.")

    totals = split_data.get("totals", {})

    if totals.get("train_count") != config["training"]["expected_train_episodes"]:
        raise ValueError(
            f"Split train_count does not match expected_train_episodes: "
            f"{totals.get('train_count')} vs {config['training']['expected_train_episodes']}"
        )

    check_openvla_split_patch(config)

    return split_data


# This function saves the resolved config next to the adapter output folder.
def save_resolved_config(config: dict) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    resolved_config_path = output_dir / "resolved_df_04_01_lora_train_config.yaml"

    with resolved_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


# This function builds the official OpenVLA LoRA fine-tuning command.
def build_finetune_command(config: dict) -> List[str]:
    output_dir = Path(config["paths"]["output_dir"])
    adapter_tmp_dir = output_dir / "adapter_tmp"

    return [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        "1",
        config["paths"]["official_finetune_py"],
        "--vla_path",
        config["model"]["model_id"],
        "--data_root_dir",
        config["paths"]["dataset_root"],
        "--dataset_name",
        config["dataset"]["mixture_name"],
        "--run_root_dir",
        str(output_dir),
        "--adapter_tmp_dir",
        str(adapter_tmp_dir),
        "--batch_size",
        str(config["training"]["batch_size"]),
        "--max_steps",
        str(config["training"]["max_steps"]),
        "--save_steps",
        str(config["training"]["save_steps"]),
        "--learning_rate",
        str(config["training"]["learning_rate"]),
        "--grad_accumulation_steps",
        str(config["training"]["grad_accumulation_steps"]),
        "--image_aug",
        str(config["training"]["image_aug"]),
        "--shuffle_buffer_size",
        str(config["training"]["shuffle_buffer_size"]),
        "--save_latest_checkpoint_only",
        str(config["training"]["save_latest_checkpoint_only"]),
        "--use_lora",
        "True",
        "--lora_rank",
        str(config["lora"]["rank"]),
        "--lora_dropout",
        str(config["lora"]["dropout"]),
        "--use_quantization",
        str(config["training"].get("use_quantization", False)),
        "--wandb_project",
        str(config["logging"]["wandb_project"]),
        "--run_id_note",
        str(config["run"]["run_id"]),
    ]


# This function tries to parse useful training metrics from the official stdout log.
def parse_training_stdout(stdout_log_path: Path) -> dict:
    metrics = {
        "total_training_steps": None,
        "final_train_loss": None,
        "best_val_loss": None,
        "best_checkpoint_step": None,
        "gradient_norm": None,
        "train_loss_curve": [],
        "val_loss_curve": [],
        "learning_rate_curve": [],
        "trainable_params": None,
    }

    if not stdout_log_path.exists():
        return metrics

    text = stdout_log_path.read_text(encoding="utf-8", errors="ignore")

    step_loss_patterns = [
        re.compile(r"step[=\s:]+(\d+).*?loss[=\s:]+([0-9.]+)", re.IGNORECASE),
        re.compile(r"loss[=\s:]+([0-9.]+).*?step[=\s:]+(\d+)", re.IGNORECASE),
    ]

    for line in text.splitlines():
        line_lower = line.lower()

        for pattern in step_loss_patterns:
            match = pattern.search(line)
            if match:
                groups = match.groups()

                if "step" in line_lower and "loss" in line_lower:
                    try:
                        if groups[0].isdigit():
                            step = int(groups[0])
                            loss = float(groups[1])
                        else:
                            loss = float(groups[0])
                            step = int(groups[1])

                        metrics["train_loss_curve"].append({"step": step, "loss": loss})
                        metrics["total_training_steps"] = max(metrics["total_training_steps"] or 0, step)
                        metrics["final_train_loss"] = loss
                    except Exception:
                        pass

        val_match = re.search(r"val.*?loss[=\s:]+([0-9.]+)", line, re.IGNORECASE)
        if val_match:
            try:
                val_loss = float(val_match.group(1))
                metrics["val_loss_curve"].append({"line": line.strip(), "val_loss": val_loss})

                if metrics["best_val_loss"] is None or val_loss < metrics["best_val_loss"]:
                    metrics["best_val_loss"] = val_loss
                    step_match = re.search(r"step[=\s:]+(\d+)", line, re.IGNORECASE)
                    if step_match:
                        metrics["best_checkpoint_step"] = int(step_match.group(1))
            except Exception:
                pass

        grad_match = re.search(r"grad(?:ient)?[_\s-]*norm[=\s:]+([0-9.]+)", line, re.IGNORECASE)
        if grad_match:
            try:
                metrics["gradient_norm"] = float(grad_match.group(1))
            except Exception:
                pass

        lr_match = re.search(r"(?:lr|learning_rate)[=\s:]+([0-9.eE-]+)", line, re.IGNORECASE)
        step_match = re.search(r"step[=\s:]+(\d+)", line, re.IGNORECASE)
        if lr_match:
            try:
                item = {"learning_rate": float(lr_match.group(1))}
                if step_match:
                    item["step"] = int(step_match.group(1))
                metrics["learning_rate_curve"].append(item)
            except Exception:
                pass

        params_match = re.search(r"trainable.*?params.*?([0-9,]+)", line, re.IGNORECASE)
        if params_match:
            metrics["trainable_params"] = params_match.group(1).replace(",", "")

    return metrics


# This function runs the official OpenVLA LoRA trainer through subprocess and streams logs.
def run_lora_training(config: dict) -> Dict[str, Any]:
    command = build_finetune_command(config)

    env = os.environ.copy()

    # Make the bundled src/prismatic package available to finetune.py.
    openvla_src_root = Path(
        config["paths"]["official_openvla_root"]
    ).resolve()

    existing_pythonpath = env.get("PYTHONPATH", "")

    env["PYTHONPATH"] = (
        str(openvla_src_root)
        if not existing_pythonpath
        else f"{openvla_src_root}{os.pathsep}{existing_pythonpath}"
    )

    env["OPENVLA_EPISODE_SPLIT_JSON"] = config["paths"]["split_json"]

    stdout_log_path = Path(config["paths"]["log_path"]).with_suffix(".stdout.log")
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with stdout_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(command) + "\n\n")
        log_file.write(f"OPENVLA_EPISODE_SPLIT_JSON={env['OPENVLA_EPISODE_SPLIT_JSON']}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=config["paths"]["official_openvla_root"],
            text=True,
            bufsize=1,
        )

        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

        returncode = process.wait()

    training_time_hours = round((time.time() - start_time) / 3600, 4)
    parsed_metrics = parse_training_stdout(stdout_log_path)

    return {
        "returncode": returncode,
        "training_time_hours": training_time_hours,
        "training_stdout_log": str(stdout_log_path),
        "command": " ".join(command),
        **parsed_metrics,
    }


# This function builds a complete metadata object for JSON and CSV logging.
def build_metadata(
    config: dict,
    dependency_status: dict,
    gpu_info: dict,
    split_data: Optional[dict],
    status: str,
    training_result: Dict[str, Any],
    failure_reason: str,
) -> dict:
    totals = split_data.get("totals", {}) if split_data else {}

    metadata = {
        "task_id": "DF-04-01",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": config["run"]["run_id"],
        "branch_name": config["run"]["branch_name"],
        "random_seed": config["run"]["random_seed"],
        "status": status,
        "failure_reason": failure_reason,
        "model_id": config["model"]["model_id"],
        "torch_dtype": config["model"]["torch_dtype"],
        "device": config["model"]["device"],
        "dataset_name": config["dataset"]["mixture_name"],
        "split_json": config["paths"]["split_json"],
        "train_count": totals.get("train_count"),
        "val_count": totals.get("val_count"),
        "test_count": totals.get("test_count"),
        "training_time_hours": training_result.get("training_time_hours"),
        "total_training_steps": training_result.get("total_training_steps"),
        "final_train_loss": training_result.get("final_train_loss"),
        "best_val_loss": training_result.get("best_val_loss"),
        "best_checkpoint_step": training_result.get("best_checkpoint_step"),
        "gradient_norm": training_result.get("gradient_norm"),
        "train_loss_curve": training_result.get("train_loss_curve", []),
        "val_loss_curve": training_result.get("val_loss_curve", []),
        "learning_rate_curve": training_result.get("learning_rate_curve", []),
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"].get("epochs"),
        "max_steps": config["training"]["max_steps"],
        "save_steps": config["training"]["save_steps"],
        "grad_accumulation_steps": config["training"]["grad_accumulation_steps"],
        "trainable_params": training_result.get("trainable_params"),
        "lora_rank": config["lora"]["rank"],
        "lora_alpha": config["lora"]["alpha"],
        "lora_dropout": config["lora"]["dropout"],
        "target_modules": config["lora"]["target_modules"],
        "adapter_path": config["paths"]["output_dir"],
        "metadata_json": config["paths"]["metadata_json"],
        "training_stdout_log": training_result.get("training_stdout_log"),
        "returncode": training_result.get("returncode"),
        "gpu_info": gpu_info,
        "dependency_status": dependency_status,
        "official_openvla_root": config["paths"]["official_openvla_root"],
        "official_finetune_py": config["paths"]["official_finetune_py"],
        "official_split_patch": "OPENVLA_EPISODE_SPLIT_JSON",
        "logging_interval_episodes": config["logging"]["print_every_episodes"],
        "notes": config.get("notes", {}),
    }

    return metadata


# This function writes metadata JSON with all required experiment fields.
def write_metadata_json(config: dict, metadata: dict) -> None:
    metadata_json = Path(config["paths"]["metadata_json"])

    with metadata_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


# This function writes a Markdown log for the DF-04-01 LoRA training run.
def write_markdown_log(config: dict, metadata: dict) -> None:
    log_path = Path(config["paths"]["log_path"])

    lines = [
        "# DF-04-01 LoRA Training Log",
        "",
        "## Notes",
        "",
        "The following paths are local paths on the author's cloud machine.",
        "They are recorded for experiment reproducibility and do not contain access credentials.",
        "",
        "## Status",
        "",
        f"- status: {metadata['status']}",
        f"- failure_reason: {metadata['failure_reason']}",
        f"- returncode: {metadata['returncode']}",
        "",
        "## Required Training Fields",
        "",
        f"- training_time_hours: {metadata['training_time_hours']}",
        f"- total_training_steps: {metadata['total_training_steps']}",
        f"- final_train_loss: {metadata['final_train_loss']}",
        f"- best_val_loss: {metadata['best_val_loss']}",
        f"- best_checkpoint_step: {metadata['best_checkpoint_step']}",
        f"- gradient_norm: {metadata['gradient_norm']}",
        f"- train_loss_curve_points: {len(metadata['train_loss_curve'])}",
        f"- val_loss_curve_points: {len(metadata['val_loss_curve'])}",
        f"- learning_rate: {metadata['learning_rate']}",
        f"- batch_size: {metadata['batch_size']}",
        f"- epochs: {metadata['epochs']}",
        f"- trainable_params: {metadata['trainable_params']}",
        f"- lora_rank: {metadata['lora_rank']}",
        f"- lora_alpha: {metadata['lora_alpha']}",
        f"- adapter_path: {metadata['adapter_path']}",
        f"- metadata_json: {metadata['metadata_json']}",
        "",
        "## Dataset Split",
        "",
        f"- dataset_name: {metadata['dataset_name']}",
        f"- split_json: {metadata['split_json']}",
        f"- train_count: {metadata['train_count']}",
        f"- val_count: {metadata['val_count']}",
        f"- test_count: {metadata['test_count']}",
        "",
        "## GPU Info",
        "",
        f"- cuda_available: {metadata['gpu_info']['cuda_available']}",
        f"- gpu_name: {metadata['gpu_info']['gpu_name']}",
        f"- gpu_count: {metadata['gpu_info']['gpu_count']}",
        f"- torch_version: {metadata['gpu_info']['torch_version']}",
        f"- cuda_version: {metadata['gpu_info']['cuda_version']}",
        f"- peak_vram_allocated_gb: {metadata['gpu_info']['peak_vram_allocated_gb']}",
        f"- peak_vram_reserved_gb: {metadata['gpu_info']['peak_vram_reserved_gb']}",
        "",
        "## Command",
        "",
        "```bash",
        metadata.get("command", ""),
        "```",
        "",
        "## Training stdout log",
        "",
        f"- {metadata['training_stdout_log']}",
        "",
    ]

    log_path.write_text("\n".join(lines), encoding="utf-8")


# This function writes a lightweight CSV summary for Git tracking.
def write_summary_csv(config: dict, metadata: dict) -> None:
    summary_csv = Path(config["paths"]["summary_csv"])

    row = {
        "run_id": metadata["run_id"],
        "branch_name": metadata["branch_name"],
        "random_seed": metadata["random_seed"],
        "status": metadata["status"],
        "failure_reason": metadata["failure_reason"],
        "model_id": metadata["model_id"],
        "torch_dtype": metadata["torch_dtype"],
        "device": metadata["device"],
        "dataset_name": metadata["dataset_name"],
        "split_json": metadata["split_json"],
        "train_count": metadata["train_count"],
        "val_count": metadata["val_count"],
        "test_count": metadata["test_count"],
        "training_time_hours": metadata["training_time_hours"],
        "total_training_steps": metadata["total_training_steps"],
        "final_train_loss": metadata["final_train_loss"],
        "best_val_loss": metadata["best_val_loss"],
        "best_checkpoint_step": metadata["best_checkpoint_step"],
        "gradient_norm": metadata["gradient_norm"],
        "learning_rate": metadata["learning_rate"],
        "batch_size": metadata["batch_size"],
        "epochs": metadata["epochs"],
        "trainable_params": metadata["trainable_params"],
        "lora_rank": metadata["lora_rank"],
        "lora_alpha": metadata["lora_alpha"],
        "adapter_path": metadata["adapter_path"],
        "metadata_json": metadata["metadata_json"],
        "gpu_name": metadata["gpu_info"]["gpu_name"],
        "torch_version": metadata["gpu_info"]["torch_version"],
        "cuda_version": metadata["gpu_info"]["cuda_version"],
        "returncode": metadata["returncode"],
        "training_stdout_log": metadata["training_stdout_log"],
    }

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# This function controls the DF-04-01 LoRA training workflow.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml"),
        help="Path to the DF-04-01 LoRA YAML config.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run checks and logging without starting full training.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    prepare_output_dirs(config)
    set_random_seed(int(config["run"]["random_seed"]))

    dependency_status = check_dependencies()
    gpu_info = get_gpu_info()

    split_data: Optional[dict] = None
    training_result: Dict[str, Any] = {}
    status = "failed"
    failure_reason = ""

    try:
        split_data = run_preflight_checks(config)

        if dependency_status.get("peft") != "installed":
            raise ImportError("peft is not installed. Install peft before starting LoRA training.")

        if args.preflight_only:
            status = "preflight_passed"
            failure_reason = ""
        else:
            training_result = run_lora_training(config)

            if training_result.get("returncode") != 0:
                raise RuntimeError(f"Official OpenVLA finetune.py failed with returncode {training_result.get('returncode')}")

            status = "training_finished"
            failure_reason = ""

    except Exception as error:
        status = "failed"
        failure_reason = str(error)

    metadata = build_metadata(
        config=config,
        dependency_status=dependency_status,
        gpu_info=gpu_info,
        split_data=split_data,
        status=status,
        training_result=training_result,
        failure_reason=failure_reason,
    )

    metadata["command"] = training_result.get("command", "")

    save_resolved_config(config)
    write_metadata_json(config, metadata)
    write_markdown_log(config, metadata)
    write_summary_csv(config, metadata)

    print(f"DF-04-01 LoRA script finished with status: {status}")
    print(f"Log path: {config['paths']['log_path']}")
    print(f"Summary CSV: {config['paths']['summary_csv']}")
    print(f"Metadata JSON: {config['paths']['metadata_json']}")

    if status == "failed":
        print(f"Failure reason: {failure_reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()