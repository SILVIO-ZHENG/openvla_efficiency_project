# Start the DF-05-01 closed-loop LIBERO simulation from one YAML config.

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


# Parse the simulation config path and optional validation-only mode.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DF-05-01 LIBERO simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/simulation_config.yaml"),
        help="Path to the simulation YAML file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the YAML schema without loading GPU or simulator dependencies.",
    )
    return parser.parse_args()


# Load one YAML mapping from disk.
def load_config(path: Path) -> Dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Simulation config not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The simulation config must contain a YAML mapping.")
    return config


# Validate all cross-field contracts required by LoRA-OneStepCE evaluation.
def validate_config(config: Mapping[str, Any]) -> None:
    required_sections = (
        "experiment",
        "model",
        "paths",
        "input",
        "simulation",
        "action",
        "timing",
        "video",
        "recording",
        "logging",
    )
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError("Missing config sections: " + ", ".join(missing))

    if config["model"]["policy_type"] != "lora_one_step_ce":
        raise ValueError("model.policy_type must be 'lora_one_step_ce'.")
    if config["model"]["training_objective"] != "token_cross_entropy":
        raise ValueError("DF-04-02 requires token_cross_entropy.")
    if int(config["action"]["action_dim"]) != 7:
        raise ValueError("LIBERO action_dim must be 7.")
    if int(config["action"]["action_chunk_size"]) != 1:
        raise ValueError("DF-04-02 action_chunk_size must be 1.")
    if int(config["action"]["actions_executed_per_inference"]) != 1:
        raise ValueError("DF-04-02 must execute one action per inference.")
    if int(config["action"]["future_action_window_size"]) != 0:
        raise ValueError("DF-04-02 future_action_window_size must be 0.")
    if int(config["input"]["num_images_in_input"]) != 2:
        raise ValueError("DF-04-02 requires primary and wrist images.")
    if int(config["input"]["proprio_dim"]) != 8:
        raise ValueError("DF-04-02 proprio_dim must be 8.")
    if not bool(config["input"]["use_wrist_image"]):
        raise ValueError("DF-04-02 requires the wrist image.")
    if not bool(config["input"]["use_proprio"]):
        raise ValueError("DF-04-02 requires proprio input.")
    if config["simulation"]["task_suite_name"] != "libero_10":
        raise ValueError("DF-05-01 evaluates the libero_10 task suite.")

    task_ids = list(config["simulation"]["task_ids"])
    if len(task_ids) != int(config["simulation"]["num_tasks"]):
        raise ValueError("simulation.num_tasks does not match task_ids.")
    if len(set(int(value) for value in task_ids)) != len(task_ids):
        raise ValueError("simulation.task_ids contains duplicates.")
    if int(config["simulation"]["num_trials_per_task"]) <= 0:
        raise ValueError("num_trials_per_task must be positive.")
    if int(config["simulation"]["max_steps_per_episode"]) <= 0:
        raise ValueError("max_steps_per_episode must be positive.")

    crop_scale = float(config["input"]["center_crop_scale"])
    if not 0.0 < crop_scale <= 1.0:
        raise ValueError("input.center_crop_scale must be in (0, 1].")
    if len(str(config["video"]["codec"])) != 4:
        raise ValueError("video.codec must contain exactly four characters.")
    if len(config["action"]["action_min"]) != 7 or len(
        config["action"]["action_max"]
    ) != 7:
        raise ValueError("action_min and action_max must each contain 7 values.")
    if not bool(config["action"]["stop_on_invalid_action"]):
        raise ValueError("DF-05-01 stops each rollout after an invalid action.")

    required_timing_flags = (
        "measure_inference_time",
        "measure_end_to_end_time",
        "measure_environment_step_time",
        "measure_effective_control_frequency",
    )
    if not all(bool(config["timing"][key]) for key in required_timing_flags):
        raise ValueError("All DF-05-01 timing measurements must remain enabled.")

    required_recording_flags = (
        "save_runtime_manifest",
        "save_resolved_config",
        "save_rollout_results",
        "save_inference_events",
        "save_failure_cases",
        "save_task_summary",
        "save_simulation_summary",
    )
    if not all(bool(config["recording"][key]) for key in required_recording_flags):
        raise ValueError("All DF-05-01 structured result files must remain enabled.")


# Resolve output paths relative to the configured cloud project root.
def resolve_output_paths(config: Dict[str, Any]) -> None:
    project_root = Path(config["paths"]["project_root"]).expanduser().resolve()
    config["paths"]["project_root"] = str(project_root)
    for key in (
        "output_root",
        "log_directory",
        "metrics_directory",
        "video_directory",
    ):
        path = Path(config["paths"][key]).expanduser()
        config["paths"][key] = str(path if path.is_absolute() else project_root / path)
    log_path = Path(config["logging"]["log_file"]).expanduser()
    config["logging"]["log_file"] = str(
        log_path if log_path.is_absolute() else project_root / log_path
    )


# Configure Python paths and headless runtime variables before simulator imports.
def configure_runtime_environment(config: Mapping[str, Any]) -> None:
    project_root = str(config["paths"]["project_root"])
    source_root = str(Path(config["paths"]["openvla_source_root"]).expanduser())
    libero_root = str(Path(config["paths"]["libero_root"]).expanduser())
    sys.path[:0] = [project_root, source_root, libero_root]

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [project_root, source_root, libero_root]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


# Create output directories and configure console plus file logging.
def configure_logging(config: Mapping[str, Any]) -> None:
    for key in ("output_root", "log_directory", "metrics_directory", "video_directory"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    log_path = Path(config["logging"]["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(config["logging"]["log_level"]).upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


# Seed Python, NumPy, PyTorch, and CUDA reproducibly.
def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Save the fully resolved YAML used by the run.
def save_resolved_config(config: Mapping[str, Any]) -> None:
    path = Path(config["paths"]["metrics_directory"]) / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)


# Validate configuration, initialize components, and run all rollouts.
def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        validate_config(config)
        if args.validate_only:
            print("DF-05-01 config validation: OK")
            return 0

        resolve_output_paths(config)
        configure_runtime_environment(config)
        configure_logging(config)
        seed_everything(int(config["experiment"]["seed"]))
        if bool(config["recording"]["save_resolved_config"]):
            save_resolved_config(config)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("DF-05-01 requires a CUDA GPU.")

        # Delay imports until PYTHONPATH and EGL are configured.
        from src.libero_env import LiberoEnvironment
        from src.lora_policy import LoraPolicy
        from src.simulation_runner import SimulationRunner

        logging.info("Loading DF-04-02 best adapter from step 10000.")
        policy = LoraPolicy(config)
        environment = LiberoEnvironment(config)
        runner = SimulationRunner(config, policy, environment)
        summary = runner.run()
        logging.info(
            "Simulation finished: rollouts=%s success_rate=%s completed=%s",
            summary["total_rollouts"],
            summary["overall_success_rate"],
            summary["completed"],
        )
        return 0
    except KeyboardInterrupt:
        logging.warning("Simulation interrupted by the user.")
        return 130
    except Exception:
        logging.exception("DF-05-01 simulation failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
