# Start the DF-05-02 LoRA+OFT closed-loop LIBERO simulation.

import argparse
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


# Parse the YAML path and optional validation or preflight mode.
def parse_args() -> argparse.Namespace:
    # Keep local validation independent from CUDA and LIBERO imports.
    parser = argparse.ArgumentParser(description="Run DF-05-02 LIBERO simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/simulation_config.yaml"),
        help="Path to the simulation YAML file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the YAML schema without loading GPU dependencies.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load all components and predict one action chunk without executing it.",
    )
    return parser.parse_args()


# Load one YAML mapping from disk.
def load_config(path: Path) -> Dict[str, Any]:
    # Resolve the path before opening it so records use an unambiguous location.
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Simulation config not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The simulation config must contain a YAML mapping.")
    return config


# Validate the fixed LoRA+OFT evaluation contract and all cross-field settings.
def validate_config(config: Mapping[str, Any]) -> None:
    # Reject incomplete configs before any model or simulator dependency is loaded.
    required_sections = (
        "experiment",
        "model",
        "paths",
        "runtime",
        "training_contract",
        "input",
        "simulation",
        "action",
        "collapse_diagnostics",
        "timing",
        "video",
        "recording",
        "logging",
    )
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError("Missing config sections: " + ", ".join(missing))

    model = config["model"]
    action = config["action"]
    input_config = config["input"]
    simulation = config["simulation"]
    diagnostics = config["collapse_diagnostics"]

    if model["policy_type"] != "lora_oft_l1_chunk":
        raise ValueError("model.policy_type must be 'lora_oft_l1_chunk'.")
    if model["training_objective"] != "l1_regression":
        raise ValueError("DF-04-03 requires l1_regression.")
    if model["action_representation"] != "continuous_actions":
        raise ValueError("DF-04-03 requires continuous_actions.")
    if not bool(model["use_l1_regression"]) or not bool(model["use_action_head"]):
        raise ValueError("DF-04-03 requires the L1 continuous action head.")
    if bool(model["use_diffusion"]) or bool(model["use_film"]):
        raise ValueError("This DF-04-03 run uses neither diffusion nor FiLM.")
    if not bool(model["strict_continuous_action_path"]):
        raise ValueError("The strict continuous action path must remain enabled.")
    if not bool(model["forbid_discrete_action_fallback"]):
        raise ValueError("Discrete action fallback must remain forbidden.")
    if str(model["attn_implementation"]) != "eager":
        raise ValueError("DF-05-02 must use eager attention for fair comparison.")
    if int(model["checkpoint_step"]) != 100000:
        raise ValueError("DF-05-02 must evaluate the step-100000 best adapter.")

    if int(action["action_dim"]) != 7:
        raise ValueError("LIBERO action_dim must be 7.")
    if int(action["action_chunk_size"]) != 8:
        raise ValueError("DF-04-03 action_chunk_size must be 8.")
    if int(action["actions_executed_per_inference"]) != 8:
        raise ValueError("Official OpenVLA-OFT evaluation executes all 8 actions.")
    if int(action["future_action_window_size"]) != 7:
        raise ValueError("An 8-step chunk requires future_action_window_size=7.")
    if len(action["action_min"]) != 7 or len(action["action_max"]) != 7:
        raise ValueError("action_min and action_max must each contain 7 values.")
    if not bool(action["stop_on_invalid_action"]):
        raise ValueError("DF-05-02 must stop a rollout after an invalid action.")

    if int(input_config["num_images_in_input"]) != 2:
        raise ValueError("DF-04-03 requires primary and wrist images.")
    if int(input_config["proprio_dim"]) != 8:
        raise ValueError("DF-04-03 proprio_dim must be 8.")
    if not bool(input_config["use_wrist_image"]) or not bool(
        input_config["use_proprio"]
    ):
        raise ValueError("DF-04-03 requires wrist image and proprio inputs.")
    crop_scale = float(input_config["center_crop_scale"])
    if not 0.0 < crop_scale <= 1.0:
        raise ValueError("input.center_crop_scale must be in (0, 1].")

    if simulation["task_suite_name"] != "libero_10":
        raise ValueError("DF-05-02 evaluates the libero_10 task suite.")
    task_ids = [int(value) for value in simulation["task_ids"]]
    if len(task_ids) != int(simulation["num_tasks"]):
        raise ValueError("simulation.num_tasks does not match task_ids.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("simulation.task_ids contains duplicates.")
    if int(simulation["num_trials_per_task"]) <= 0:
        raise ValueError("num_trials_per_task must be positive.")
    if int(simulation["max_steps_per_episode"]) <= 0:
        raise ValueError("max_steps_per_episode must be positive.")

    if not bool(diagnostics["enabled"]) or not bool(diagnostics["diagnostic_only"]):
        raise ValueError("Collapse diagnostics must remain enabled and diagnostic-only.")
    if bool(diagnostics["terminate_on_collapse_warning"]):
        raise ValueError("Collapse warnings cannot terminate a formal rollout.")
    if float(diagnostics["repeated_action_atol"]) < 0.0:
        raise ValueError("repeated_action_atol cannot be negative.")
    if float(diagnostics["repeated_chunk_atol"]) < 0.0:
        raise ValueError("repeated_chunk_atol cannot be negative.")

    if len(str(config["video"]["codec"])) != 4:
        raise ValueError("video.codec must contain exactly four characters.")
    required_timing_flags = (
        "measure_inference_time",
        "measure_end_to_end_time",
        "measure_environment_step_time",
        "measure_effective_control_frequency",
    )
    if not all(bool(config["timing"][key]) for key in required_timing_flags):
        raise ValueError("All structured timing measurements must remain enabled.")
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
        raise ValueError("All structured result files must remain enabled.")


# Resolve every output path relative to the configured cloud project root.
def resolve_output_paths(config: Dict[str, Any]) -> None:
    # Preserve model artifact paths while resolving only run-owned output paths.
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


# Configure import paths, EGL, CPU threading, and offline model loading.
def configure_runtime_environment(config: Mapping[str, Any]) -> None:
    # Prepend the exact DF-04-03 source before importing any prismatic module.
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


# Create all output directories and configure console plus file logging.
def configure_logging(config: Mapping[str, Any]) -> None:
    # Create directories before constructing the file handler.
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
    # Import GPU libraries only after runtime paths and environment variables are set.
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Save the fully resolved YAML used by the current invocation.
def save_resolved_config(config: Mapping[str, Any]) -> None:
    # Write the exact paths and values that downstream records should cite.
    path = Path(config["paths"]["metrics_directory"]) / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)


# Return the active Git branch for one repository path.
def get_git_branch(project_root: str) -> str:
    # Resolve Git state without mutating the repository.
    result = subprocess.run(
        ["git", "-C", project_root, "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


# Validate the actual Python, GPU, and Git branch used for simulation.
def validate_runtime_identity(config: Mapping[str, Any]) -> None:
    # Fail before model loading when the run would not be comparable to DF-05-01.
    import torch

    expected_python = str(config["runtime"]["expected_python_major_minor"])
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise RuntimeError(
            f"Expected Python {expected_python}, found {actual_python}."
        )

    gpu_name = torch.cuda.get_device_name(torch.device(config["model"]["device"]))
    expected_gpu = str(config["runtime"]["expected_gpu_name"])
    if bool(config["runtime"]["enforce_expected_gpu"]) and gpu_name != expected_gpu:
        raise RuntimeError(f"Expected GPU {expected_gpu!r}, found {gpu_name!r}.")

    branch = get_git_branch(str(config["paths"]["project_root"]))
    expected_branch = str(config["runtime"]["expected_git_branch"])
    if bool(config["runtime"]["enforce_expected_git_branch"]) and branch != expected_branch:
        raise RuntimeError(
            f"Expected Git branch {expected_branch!r}, found {branch!r}."
        )


# Load one task observation and verify the complete model prediction path.
def run_preflight(
    config: Mapping[str, Any],
    policy: Any,
    environment: Any,
) -> None:
    # Use the first official initial state without executing a policy action.
    task_index = int(config["simulation"]["task_ids"][0])
    task_info = environment.prepare_task(task_index)
    observation = environment.reset(task_info["initial_states"][0])
    for _ in range(int(config["simulation"]["initial_wait_steps"])):
        observation, _, _, _ = environment.step(environment.dummy_action())
    observation_data = environment.extract_observation(observation)
    chunk, info = policy.predict(
        primary_image=observation_data["primary_image"],
        wrist_image=observation_data["wrist_image"],
        proprio=observation_data["proprio"],
        instruction=str(task_info["task_instruction"]),
    )
    logging.info(
        "DF-05-02 preflight: action_shape=%s latency_ms=%.3f zero_equivalent=%d",
        tuple(chunk.shape),
        float(info["inference_latency_ms"]),
        int(sum(info["zero_equivalent_arm_mask"])),
    )
    environment.close()


# Validate configuration, initialize components, and run all requested rollouts.
def main() -> int:
    # Keep one top-level exception boundary so cloud logs preserve the root failure.
    args = parse_args()
    try:
        config = load_config(args.config)
        validate_config(config)
        if args.validate_only:
            print("DF-05-02 config validation: OK")
            return 0

        resolve_output_paths(config)
        configure_runtime_environment(config)
        configure_logging(config)
        seed_everything(int(config["experiment"]["seed"]))
        if bool(config["recording"]["save_resolved_config"]):
            save_resolved_config(config)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("DF-05-02 requires a CUDA GPU.")
        validate_runtime_identity(config)

        # Delay project imports until the exact source roots and EGL are configured.
        from src.libero_env import LiberoEnvironment
        from src.lora_oft_policy import LoraOFTPolicy
        from src.simulation_runner import SimulationRunner

        logging.info("Loading DF-04-03 best adapter from step 100000.")
        policy = LoraOFTPolicy(config)
        environment = LiberoEnvironment(config)
        if args.preflight_only:
            run_preflight(config, policy, environment)
            return 0

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
        logging.exception("DF-05-02 simulation failed.")
        return 1


# Exit with the main return code when executed as a script.
if __name__ == "__main__":
    raise SystemExit(main())
