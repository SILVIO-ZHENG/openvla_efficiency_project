# Run closed-loop LIBERO rollouts and write reproducible evaluation records.

import csv
import importlib.metadata
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from src.libero_env import LiberoEnvironment
from src.lora_policy import LoraPolicy


# Define the stable per-rollout CSV schema.
ROLLOUT_FIELDS = [
    "rollout_id",
    "run_id",
    "model_variant",
    "task_index",
    "task_name",
    "task_instruction",
    "trial_index",
    "initial_state_id",
    "rollout_seed",
    "success",
    "timeout",
    "termination_reason",
    "episode_steps",
    "steps_to_success",
    "episode_time_seconds",
    "inference_calls",
    "actions_generated",
    "actions_executed",
    "replan_count",
    "avg_inference_latency_ms",
    "p95_inference_latency_ms",
    "avg_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "control_frequency_hz",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
    "invalid_action_count",
    "automatic_failure_reason",
    "failure_detail",
    "manual_failure_reason",
    "manual_failure_note",
    "video_path",
    "started_at",
    "finished_at",
]


# Define the stable failed-rollout CSV schema.
FAILURE_FIELDS = [
    "rollout_id",
    "model_variant",
    "task_name",
    "trial_index",
    "automatic_failure_reason",
    "failure_detail",
    "manual_failure_reason",
    "manual_failure_note",
    "episode_steps",
    "last_action",
    "video_path",
    "timestamp",
]


# Return the current UTC timestamp in ISO-8601 format.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Return a float mean or None when no values are available.
def optional_mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


# Return one percentile or None when no values are available.
def optional_percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    return float(np.percentile(values, percentile)) if values else None


# Write one JSON document atomically.
def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


# Execute rollouts and record all metrics required by DF-05-01.
class SimulationRunner:
    # Initialize output paths and resume state.
    def __init__(
        self,
        config: Mapping[str, Any],
        policy: LoraPolicy,
        environment: LiberoEnvironment,
    ) -> None:
        self.config = config
        self.policy = policy
        self.environment = environment
        self.experiment_config = config["experiment"]
        self.model_config = config["model"]
        self.path_config = config["paths"]
        self.simulation_config = config["simulation"]
        self.action_config = config["action"]
        self.timing_config = config["timing"]
        self.video_config = config["video"]
        self.recording_config = config["recording"]
        self.logging_config = config["logging"]

        self.metrics_directory = Path(self.path_config["metrics_directory"])
        self.video_directory = Path(self.path_config["video_directory"])
        self.rollout_results_path = self.metrics_directory / "rollout_results.csv"
        self.inference_events_path = self.metrics_directory / "inference_events.jsonl"
        self.failure_cases_path = self.metrics_directory / "failure_cases.csv"
        self.task_summary_path = self.metrics_directory / "task_summary.csv"
        self.simulation_summary_path = self.metrics_directory / "simulation_summary.json"
        self.runtime_manifest_path = self.metrics_directory / "runtime_manifest.json"
        self.started_at = utc_now()
        self.started_perf_counter = time.perf_counter()
        self.inference_handle: Optional[Any] = None

    # Append one row using a stable CSV header.
    def _append_csv(self, path: Path, fieldnames: Sequence[str], row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            if bool(self.logging_config["flush_after_each_rollout"]):
                handle.flush()
                os.fsync(handle.fileno())

    # Read all rollout rows belonging to the current run.
    def _read_rollout_rows(self) -> List[Dict[str, str]]:
        if not self.rollout_results_path.is_file():
            return []
        with self.rollout_results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        run_id = str(self.experiment_config["run_id"])
        return [row for row in rows if row.get("run_id") == run_id]

    # Read inference timing events belonging to the current run.
    def _read_inference_events(self) -> List[Dict[str, Any]]:
        if not self.inference_events_path.is_file():
            return []
        run_id = str(self.experiment_config["run_id"])
        events = []
        with self.inference_events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("run_id") == run_id:
                    events.append(event)
        return events

    # Write one inference event immediately for crash-safe timing records.
    def _write_inference_event(self, event: Mapping[str, Any]) -> None:
        if self.inference_handle is None:
            raise RuntimeError("Inference event file is not open.")
        self.inference_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.inference_handle.flush()

    # Convert the model gripper value to the LIBERO action convention.
    def _postprocess_action(self, action: np.ndarray) -> np.ndarray:
        processed = np.asarray(action, dtype=np.float32).copy()
        expected_shape = (int(self.action_config["action_dim"]),)
        if processed.shape != expected_shape:
            raise ValueError(
                f"Expected action shape {expected_shape}, got {processed.shape}."
            )
        if bool(self.action_config["reject_nan_or_inf"]) and not np.isfinite(processed).all():
            raise ValueError("Action contains NaN or Inf.")

        if bool(self.action_config["convert_gripper_to_libero_convention"]):
            processed[-1] = 2.0 * processed[-1] - 1.0
            if bool(self.action_config["binarize_gripper"]):
                processed[-1] = np.sign(processed[-1])
            processed[-1] *= -1.0

        if bool(self.action_config["clip_actions"]):
            lower = np.asarray(self.action_config["action_min"], dtype=np.float32)
            upper = np.asarray(self.action_config["action_max"], dtype=np.float32)
            processed = np.clip(processed, lower, upper)
        return processed

    # Open one MP4 writer for a rollout.
    def _create_video_writer(self, video_path: Path) -> Optional[cv2.VideoWriter]:
        if not bool(self.video_config["enabled"]):
            return None
        resolution = int(self.config["input"]["environment_render_resolution"])
        width = resolution * (2 if bool(self.video_config["include_wrist_view"]) else 1)
        codec = cv2.VideoWriter_fourcc(*str(self.video_config["codec"]))
        writer = cv2.VideoWriter(
            str(video_path),
            codec,
            float(self.video_config["fps"]),
            (width, resolution),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")
        return writer

    # Write one annotated primary-and-wrist video frame.
    def _write_video_frame(
        self,
        writer: Optional[cv2.VideoWriter],
        observation_data: Mapping[str, np.ndarray],
        task_instruction: str,
        trial_index: int,
        step: int,
        inference_latency_ms: Optional[float],
        success: bool,
    ) -> None:
        if writer is None:
            return
        primary = observation_data["primary_image"]
        if bool(self.video_config["include_wrist_view"]):
            frame_rgb = np.concatenate(
                (primary, observation_data["wrist_image"]),
                axis=1,
            )
        else:
            frame_rgb = primary
        frame = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)

        lines = []
        if bool(self.video_config["show_model_name"]):
            lines.append(str(self.model_config["model_variant"]))
        if bool(self.video_config["show_task_instruction"]):
            lines.append(str(task_instruction)[:90])
        status_parts = []
        if bool(self.video_config["show_trial_index"]):
            status_parts.append(f"trial={trial_index}")
        if bool(self.video_config["show_step_number"]):
            status_parts.append(f"step={step}")
        if bool(self.video_config["show_inference_latency"]) and inference_latency_ms is not None:
            status_parts.append(f"inference={inference_latency_ms:.1f}ms")
        if bool(self.video_config["show_success_status"]):
            status_parts.append(f"success={success}")
        if status_parts:
            lines.append(" | ".join(status_parts))

        overlay_height = max(28, 24 * len(lines))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (8, 18 + 23 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        writer.write(frame)

    # Execute one complete rollout and return its result and optional failure row.
    def _run_rollout(
        self,
        task_info: Mapping[str, Any],
        trial_index: int,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], bool]:
        task_index = int(task_info["task_index"])
        rollout_id = f"task_{task_index:02d}_trial_{trial_index:03d}"
        rollout_seed = int(self.experiment_config["seed"]) + task_index * 1000 + trial_index
        started_at = utc_now()
        episode_started = time.perf_counter()
        success = False
        timeout = False
        interrupted = False
        termination_reason = ""
        failure_detail = ""
        episode_steps = 0
        inference_calls = 0
        actions_generated = 0
        actions_executed = 0
        invalid_action_count = 0
        inference_latencies: List[float] = []
        end_to_end_latencies: List[float] = []
        last_action: Optional[np.ndarray] = None

        extension = str(self.video_config["file_extension"])
        video_path = self.video_directory / f"{rollout_id}{extension}"
        writer: Optional[cv2.VideoWriter] = None

        torch.cuda.reset_peak_memory_stats(self.policy.device)
        observation = self.environment.reset(task_info["initial_states"][trial_index])

        try:
            writer = self._create_video_writer(video_path)

            # Let all movable objects settle before the first policy query.
            for wait_step in range(int(self.simulation_config["initial_wait_steps"])):
                observation, _, _, _ = self.environment.step(self.environment.dummy_action())
                observation_data = self.environment.extract_observation(observation)
                self._write_video_frame(
                    writer,
                    observation_data,
                    str(task_info["task_instruction"]),
                    trial_index,
                    wait_step - int(self.simulation_config["initial_wait_steps"]),
                    None,
                    False,
                )

            while episode_steps < int(self.simulation_config["max_steps_per_episode"]):
                cycle_started = time.perf_counter()
                observation_data = self.environment.extract_observation(observation)
                inference_calls += 1

                try:
                    action, inference_info = self.policy.predict(
                        primary_image=observation_data["primary_image"],
                        wrist_image=observation_data["wrist_image"],
                        proprio=observation_data["proprio"],
                        instruction=str(task_info["task_instruction"]),
                    )
                    inference_latency_ms = float(inference_info["inference_latency_ms"])
                    inference_latencies.append(inference_latency_ms)
                    actions_generated += int(inference_info["returned_chunk_size"])
                    action = self._postprocess_action(action)
                except Exception as error:
                    invalid_action_count += 1
                    failure_detail = f"{type(error).__name__}: {error}"
                    if "shape" in str(error).lower():
                        termination_reason = "invalid_action_shape"
                    elif "nan" in str(error).lower() or "inf" in str(error).lower():
                        termination_reason = "nan_or_inf_action"
                    elif "decode" in str(error).lower():
                        termination_reason = "action_decode_error"
                    else:
                        termination_reason = "model_inference_error"
                    self._write_inference_event(
                        {
                            "run_id": self.experiment_config["run_id"],
                            "rollout_id": rollout_id,
                            "task_index": task_index,
                            "inference_index": inference_calls,
                            "environment_step": episode_steps,
                            "returned_action_shape": None,
                            "returned_chunk_size": 0,
                            "executed_chunk_size": 0,
                            "inference_latency_ms": None,
                            "amortized_inference_ms_per_action": None,
                            "environment_step_latency_ms": None,
                            "end_to_end_latency_ms": None,
                            "invalid_action": True,
                            "nan_or_inf_action": "NaN" in str(error) or "Inf" in str(error),
                            "error_message": failure_detail,
                            "timestamp": utc_now(),
                        }
                    )
                    break

                environment_step_started = time.perf_counter()
                try:
                    observation, _, done, _ = self.environment.step(action)
                except Exception as error:
                    termination_reason = "environment_error"
                    failure_detail = f"{type(error).__name__}: {error}"
                    self._write_inference_event(
                        {
                            "run_id": self.experiment_config["run_id"],
                            "rollout_id": rollout_id,
                            "task_index": task_index,
                            "inference_index": inference_calls,
                            "environment_step": episode_steps,
                            "returned_action_shape": inference_info["returned_action_shape"],
                            "returned_chunk_size": inference_info["returned_chunk_size"],
                            "executed_chunk_size": 0,
                            "inference_latency_ms": inference_latency_ms,
                            "amortized_inference_ms_per_action": None,
                            "environment_step_latency_ms": (
                                time.perf_counter() - environment_step_started
                            )
                            * 1000.0,
                            "end_to_end_latency_ms": None,
                            "invalid_action": False,
                            "nan_or_inf_action": False,
                            "error_message": failure_detail,
                            "timestamp": utc_now(),
                        }
                    )
                    break
                environment_step_latency_ms = (
                    time.perf_counter() - environment_step_started
                ) * 1000.0
                episode_steps += 1
                actions_executed += 1
                last_action = action.copy()

                if bool(self.logging_config["print_action_shape"]):
                    logging.info("%s action_shape=%s", rollout_id, action.shape)
                if bool(self.logging_config["print_action_values"]):
                    logging.info("%s action=%s", rollout_id, action.tolist())

                if bool(self.timing_config["enforce_real_time_control"]):
                    target_period = 1.0 / float(
                        self.timing_config["target_control_frequency_hz"]
                    )
                    remaining = target_period - (time.perf_counter() - cycle_started)
                    if remaining > 0.0:
                        time.sleep(remaining)

                end_to_end_latency_ms = (time.perf_counter() - cycle_started) * 1000.0
                end_to_end_latencies.append(end_to_end_latency_ms)
                success = bool(done)

                self._write_inference_event(
                    {
                        "run_id": self.experiment_config["run_id"],
                        "rollout_id": rollout_id,
                        "task_index": task_index,
                        "inference_index": inference_calls,
                        "environment_step": episode_steps,
                        "returned_action_shape": inference_info["returned_action_shape"],
                        "returned_chunk_size": inference_info["returned_chunk_size"],
                        "executed_chunk_size": 1,
                        "inference_latency_ms": inference_latency_ms,
                        "amortized_inference_ms_per_action": inference_latency_ms,
                        "environment_step_latency_ms": environment_step_latency_ms,
                        "end_to_end_latency_ms": end_to_end_latency_ms,
                        "invalid_action": False,
                        "nan_or_inf_action": False,
                        "error_message": "",
                        "timestamp": utc_now(),
                    }
                )

                next_observation_data = self.environment.extract_observation(observation)
                self._write_video_frame(
                    writer,
                    next_observation_data,
                    str(task_info["task_instruction"]),
                    trial_index,
                    episode_steps,
                    inference_latency_ms,
                    success,
                )

                log_interval = int(self.logging_config["log_every_environment_steps"])
                if log_interval > 0 and episode_steps % log_interval == 0:
                    logging.info(
                        "%s step=%d inference_ms=%.3f end_to_end_ms=%.3f",
                        rollout_id,
                        episode_steps,
                        inference_latency_ms,
                        end_to_end_latency_ms,
                    )

                if success and bool(self.simulation_config["stop_on_success"]):
                    termination_reason = "success"
                    break

            if not success and not termination_reason:
                timeout = episode_steps >= int(
                    self.simulation_config["max_steps_per_episode"]
                )
                termination_reason = "timeout" if timeout else "unknown_failure"
        except KeyboardInterrupt:
            interrupted = True
            termination_reason = "interrupted"
            failure_detail = "KeyboardInterrupt"
        finally:
            if writer is not None:
                writer.release()

        keep_video = bool(self.video_config["enabled"]) and (
            (success and bool(self.video_config["save_success_videos"]))
            or (not success and bool(self.video_config["save_failure_videos"]))
        )
        if not keep_video and video_path.exists():
            video_path.unlink()

        episode_time_seconds = time.perf_counter() - episode_started
        result = {
            "rollout_id": rollout_id,
            "run_id": self.experiment_config["run_id"],
            "model_variant": self.model_config["model_variant"],
            "task_index": task_index,
            "task_name": task_info["task_name"],
            "task_instruction": task_info["task_instruction"],
            "trial_index": trial_index,
            "initial_state_id": trial_index,
            "rollout_seed": rollout_seed,
            "success": success,
            "timeout": timeout,
            "termination_reason": termination_reason,
            "episode_steps": episode_steps,
            "steps_to_success": episode_steps if success else "",
            "episode_time_seconds": round(episode_time_seconds, 6),
            "inference_calls": inference_calls,
            "actions_generated": actions_generated,
            "actions_executed": actions_executed,
            "replan_count": max(0, inference_calls - 1),
            "avg_inference_latency_ms": optional_mean(inference_latencies),
            "p95_inference_latency_ms": optional_percentile(inference_latencies, 95),
            "avg_end_to_end_latency_ms": optional_mean(end_to_end_latencies),
            "p95_end_to_end_latency_ms": optional_percentile(end_to_end_latencies, 95),
            "control_frequency_hz": (
                1000.0 / optional_mean(end_to_end_latencies)
                if end_to_end_latencies and optional_mean(end_to_end_latencies)
                else None
            ),
            "peak_vram_allocated_gb": torch.cuda.max_memory_allocated(
                self.policy.device
            )
            / 1024**3,
            "peak_vram_reserved_gb": torch.cuda.max_memory_reserved(
                self.policy.device
            )
            / 1024**3,
            "invalid_action_count": invalid_action_count,
            "automatic_failure_reason": "" if success else termination_reason,
            "failure_detail": failure_detail,
            "manual_failure_reason": "",
            "manual_failure_note": "",
            "video_path": str(video_path) if keep_video else "",
            "started_at": started_at,
            "finished_at": utc_now(),
        }

        failure_row = None
        if not success:
            failure_row = {
                "rollout_id": rollout_id,
                "model_variant": self.model_config["model_variant"],
                "task_name": task_info["task_name"],
                "trial_index": trial_index,
                "automatic_failure_reason": termination_reason,
                "failure_detail": failure_detail,
                "manual_failure_reason": "",
                "manual_failure_note": "",
                "episode_steps": episode_steps,
                "last_action": json.dumps(last_action.tolist()) if last_action is not None else "",
                "video_path": str(video_path) if keep_video else "",
                "timestamp": utc_now(),
            }
        return result, failure_row, interrupted

    # Return a package version without failing when metadata is unavailable.
    @staticmethod
    def _package_version(package_name: str) -> Optional[str]:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    # Return one Git value from the project repository.
    def _git_value(self, arguments: Sequence[str]) -> Optional[str]:
        result = subprocess.run(
            ["git", "-C", self.path_config["project_root"], *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    # Build the reproducibility manifest for the current run.
    def _build_runtime_manifest(self, finished_at: Optional[str]) -> Dict[str, Any]:
        task_ids = [int(value) for value in self.simulation_config["task_ids"]]
        return {
            "task_id": self.experiment_config["task_id"],
            "run_id": self.experiment_config["run_id"],
            "model_variant": self.model_config["model_variant"],
            "model_family": self.model_config["model_family"],
            "training_run_id": self.model_config["training_run_id"],
            "checkpoint_step": self.model_config["checkpoint_step"],
            "training_objective": self.model_config["training_objective"],
            "action_representation": self.model_config["action_representation"],
            "action_chunk_size": self.action_config["action_chunk_size"],
            "actions_executed_per_inference": self.action_config[
                "actions_executed_per_inference"
            ],
            "suite_name": self.simulation_config["task_suite_name"],
            "selected_task_ids": task_ids,
            "num_tasks": len(task_ids),
            "num_trials_per_task": self.simulation_config["num_trials_per_task"],
            "max_episode_steps": self.simulation_config["max_steps_per_episode"],
            "initial_wait_steps": self.simulation_config["initial_wait_steps"],
            "seed": self.experiment_config["seed"],
            "environment_seed": self.simulation_config["environment_seed"],
            "use_wrist_image": self.config["input"]["use_wrist_image"],
            "use_proprio": self.config["input"]["use_proprio"],
            "num_images_in_input": self.config["input"]["num_images_in_input"],
            "proprio_dim": self.config["input"]["proprio_dim"],
            "torch_dtype": self.model_config["torch_dtype"],
            "device": self.model_config["device"],
            "gpu_name": torch.cuda.get_device_name(self.policy.device),
            "base_model_path": self.path_config["base_model_path"],
            "adapter_path": self.path_config["adapter_path"],
            "processor_path": self.path_config["processor_path"],
            "proprio_projector_path": self.path_config["proprio_projector_path"],
            "dataset_statistics_path": self.path_config["dataset_statistics_path"],
            "openvla_source_root": self.path_config["openvla_source_root"],
            "git_branch": self._git_value(["branch", "--show-current"]),
            "git_commit": self._git_value(["rev-parse", "HEAD"]),
            "torch_version": torch.__version__,
            "transformers_version": self._package_version("transformers"),
            "peft_version": self._package_version("peft"),
            "tensorflow_version": self._package_version("tensorflow"),
            "opencv_version": self._package_version("opencv-python"),
            "libero_version": self._package_version("libero"),
            "robosuite_version": self._package_version("robosuite"),
            "mujoco_version": self._package_version("mujoco"),
            "started_at": self.started_at,
            "finished_at": finished_at,
        }

    # Write per-task and overall summaries from raw CSV and JSONL records.
    def _generate_summaries(self, requested_completed: bool) -> Dict[str, Any]:
        rows = self._read_rollout_rows()
        events = self._read_inference_events()
        task_ids = [int(value) for value in self.simulation_config["task_ids"]]
        task_summaries = []

        for task_index in task_ids:
            task_rows = [row for row in rows if int(row["task_index"]) == task_index]
            task_rollout_ids = {row["rollout_id"] for row in task_rows}
            task_events = [
                event for event in events if event.get("rollout_id") in task_rollout_ids
            ]
            success_rows = [row for row in task_rows if row["success"].lower() == "true"]
            timeout_rows = [row for row in task_rows if row["timeout"].lower() == "true"]
            inference_values = [
                float(event["inference_latency_ms"])
                for event in task_events
                if event.get("inference_latency_ms") is not None
            ]
            end_to_end_values = [
                float(event["end_to_end_latency_ms"])
                for event in task_events
                if event.get("end_to_end_latency_ms") is not None
            ]
            task_summaries.append(
                {
                    "task_index": task_index,
                    "task_name": task_rows[0]["task_name"] if task_rows else "",
                    "total_rollouts": len(task_rows),
                    "success_count": len(success_rows),
                    "failure_count": len(task_rows) - len(success_rows),
                    "success_rate": len(success_rows) / len(task_rows) if task_rows else None,
                    "timeout_count": len(timeout_rows),
                    "timeout_rate": len(timeout_rows) / len(task_rows) if task_rows else None,
                    "avg_success_steps": optional_mean(
                        [float(row["episode_steps"]) for row in success_rows]
                    ),
                    "avg_all_steps": optional_mean(
                        [float(row["episode_steps"]) for row in task_rows]
                    ),
                    "avg_episode_time_seconds": optional_mean(
                        [float(row["episode_time_seconds"]) for row in task_rows]
                    ),
                    "avg_inference_latency_ms": optional_mean(inference_values),
                    "p95_inference_latency_ms": optional_percentile(inference_values, 95),
                    "avg_end_to_end_latency_ms": optional_mean(end_to_end_values),
                    "p95_end_to_end_latency_ms": optional_percentile(end_to_end_values, 95),
                    "avg_control_frequency_hz": (
                        1000.0 / optional_mean(end_to_end_values)
                        if end_to_end_values and optional_mean(end_to_end_values)
                        else None
                    ),
                    "peak_vram_gb": max(
                        [float(row["peak_vram_allocated_gb"]) for row in task_rows],
                        default=None,
                    ),
                }
            )

        self.task_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.task_summary_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(task_summaries[0].keys()) if task_summaries else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(task_summaries)

        success_rows = [row for row in rows if row["success"].lower() == "true"]
        timeout_rows = [row for row in rows if row["timeout"].lower() == "true"]
        inference_values = [
            float(event["inference_latency_ms"])
            for event in events
            if event.get("inference_latency_ms") is not None
        ]
        amortized_values = [
            float(event["amortized_inference_ms_per_action"])
            for event in events
            if event.get("amortized_inference_ms_per_action") is not None
        ]
        end_to_end_values = [
            float(event["end_to_end_latency_ms"])
            for event in events
            if event.get("end_to_end_latency_ms") is not None
        ]
        expected_rollouts = len(task_ids) * int(
            self.simulation_config["num_trials_per_task"]
        )
        completed = requested_completed and len(rows) == expected_rollouts
        task_success_rates = [
            float(summary["success_rate"])
            for summary in task_summaries
            if summary["success_rate"] is not None
        ]

        summary = {
            "task_id": self.experiment_config["task_id"],
            "run_id": self.experiment_config["run_id"],
            "model_variant": self.model_config["model_variant"],
            "checkpoint_step": self.model_config["checkpoint_step"],
            "total_tasks": len(task_ids),
            "total_rollouts": len(rows),
            "successful_rollouts": len(success_rows),
            "failed_rollouts": len(rows) - len(success_rows),
            "overall_success_rate": len(success_rows) / len(rows) if rows else None,
            "timeout_count": len(timeout_rows),
            "overall_timeout_rate": len(timeout_rows) / len(rows) if rows else None,
            "macro_average_task_success_rate": optional_mean(task_success_rates),
            "avg_success_steps": optional_mean(
                [float(row["episode_steps"]) for row in success_rows]
            ),
            "avg_all_steps": optional_mean(
                [float(row["episode_steps"]) for row in rows]
            ),
            "avg_episode_time_seconds": optional_mean(
                [float(row["episode_time_seconds"]) for row in rows]
            ),
            "total_inference_calls": len(events),
            "avg_inference_latency_ms": optional_mean(inference_values),
            "p95_inference_latency_ms": optional_percentile(inference_values, 95),
            "avg_amortized_inference_ms_per_action": optional_mean(amortized_values),
            "avg_end_to_end_latency_ms": optional_mean(end_to_end_values),
            "p95_end_to_end_latency_ms": optional_percentile(end_to_end_values, 95),
            "avg_control_frequency_hz": (
                1000.0 / optional_mean(end_to_end_values)
                if end_to_end_values and optional_mean(end_to_end_values)
                else None
            ),
            "peak_vram_allocated_gb": max(
                [float(row["peak_vram_allocated_gb"]) for row in rows],
                default=None,
            ),
            "peak_vram_reserved_gb": max(
                [float(row["peak_vram_reserved_gb"]) for row in rows],
                default=None,
            ),
            "total_runtime_hours": sum(
                float(row["episode_time_seconds"]) for row in rows
            )
            / 3600.0,
            "current_invocation_runtime_hours": (
                time.perf_counter() - self.started_perf_counter
            )
            / 3600.0,
            "completed": completed,
        }
        write_json(self.simulation_summary_path, summary)
        return summary

    # Run every configured task and trial, with resume support.
    def run(self) -> Dict[str, Any]:
        if bool(self.recording_config["save_runtime_manifest"]):
            write_json(
                self.runtime_manifest_path,
                self._build_runtime_manifest(finished_at=None),
            )

        existing_rows = self._read_rollout_rows()
        completed_rollout_ids = {row["rollout_id"] for row in existing_rows}
        requested_completed = False
        caught_error: Optional[BaseException] = None
        self.inference_events_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.inference_handle = self.inference_events_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            for task_index in [int(value) for value in self.simulation_config["task_ids"]]:
                task_info = self.environment.prepare_task(task_index)
                trials = int(self.simulation_config["num_trials_per_task"])
                if len(task_info["initial_states"]) < trials:
                    raise ValueError(
                        f"Task {task_index} has {len(task_info['initial_states'])} initial states, "
                        f"but {trials} trials were requested."
                    )
                if bool(self.logging_config["print_task_instruction"]):
                    logging.info(
                        "Starting task=%d name=%s instruction=%s",
                        task_index,
                        task_info["task_name"],
                        task_info["task_instruction"],
                    )
                else:
                    logging.info(
                        "Starting task=%d name=%s",
                        task_index,
                        task_info["task_name"],
                    )

                for trial_index in range(trials):
                    rollout_id = f"task_{task_index:02d}_trial_{trial_index:03d}"
                    if (
                        bool(self.simulation_config["resume_from_existing_results"])
                        and bool(self.simulation_config["skip_completed_rollouts"])
                        and rollout_id in completed_rollout_ids
                    ):
                        logging.info("Skipping completed rollout=%s", rollout_id)
                        continue

                    result, failure_row, interrupted = self._run_rollout(
                        task_info,
                        trial_index,
                    )
                    self._append_csv(
                        self.rollout_results_path,
                        ROLLOUT_FIELDS,
                        result,
                    )
                    if failure_row is not None:
                        self._append_csv(
                            self.failure_cases_path,
                            FAILURE_FIELDS,
                            failure_row,
                        )
                    completed_rollout_ids.add(rollout_id)
                    logging.info(
                        "Finished rollout=%s success=%s steps=%d reason=%s",
                        rollout_id,
                        result["success"],
                        result["episode_steps"],
                        result["termination_reason"],
                    )
                    if interrupted:
                        raise KeyboardInterrupt
            requested_completed = True
        except BaseException as error:
            caught_error = error
        finally:
            if self.inference_handle is not None:
                self.inference_handle.close()
                self.inference_handle = None
            self.environment.close()

        summary = self._generate_summaries(requested_completed)
        if bool(self.recording_config["save_runtime_manifest"]):
            write_json(
                self.runtime_manifest_path,
                self._build_runtime_manifest(finished_at=utc_now()),
            )
        if caught_error is not None:
            raise caught_error
        return summary
