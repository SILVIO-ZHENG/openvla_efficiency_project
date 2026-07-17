# Run chunked LIBERO rollouts and write reproducible DF-05-02 records.

import csv
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from src.libero_env import LiberoEnvironment
from src.lora_oft_policy import LoraOFTPolicy


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
    "avg_amortized_inference_ms_per_action",
    "avg_environment_step_latency_ms",
    "p95_environment_step_latency_ms",
    "avg_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "avg_amortized_end_to_end_ms_per_action",
    "control_frequency_hz",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
    "invalid_action_count",
    "zero_equivalent_arm_action_count",
    "zero_equivalent_full_noop_count",
    "repeated_action_count",
    "repeated_chunk_count",
    "max_consecutive_zero_equivalent_actions",
    "max_consecutive_repeated_chunks",
    "collapse_warning_count",
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
    "zero_equivalent_arm_action_count",
    "zero_equivalent_full_noop_count",
    "repeated_action_count",
    "repeated_chunk_count",
    "last_action",
    "video_path",
    "timestamp",
]


# Return the current UTC timestamp in ISO-8601 format.
def utc_now() -> str:
    # Use timezone-aware timestamps in every output file.
    return datetime.now(timezone.utc).isoformat()


# Return a float mean or None when no values are available.
def optional_mean(values: Sequence[float]) -> Optional[float]:
    # Preserve null for metrics that could not be measured.
    return float(np.mean(values)) if values else None


# Return one percentile or None when no values are available.
def optional_percentile(
    values: Sequence[float],
    percentile: float,
) -> Optional[float]:
    # Use NumPy's deterministic percentile implementation.
    return float(np.percentile(values, percentile)) if values else None


# Return an integer from a CSV value, treating an empty value as zero.
def csv_int(value: Any) -> int:
    # Normalize strings and numeric values emitted by csv.DictReader.
    return int(float(value)) if str(value).strip() else 0


# Write one JSON document atomically.
def write_json(path: Path, document: Mapping[str, Any]) -> None:
    # Replace the final file only after a complete temporary write.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


# Execute LoRA+OFT rollouts and record chunk-aware metrics.
class SimulationRunner:
    # Initialize output paths, configs, and invocation state.
    def __init__(
        self,
        config: Mapping[str, Any],
        policy: LoraOFTPolicy,
        environment: LiberoEnvironment,
    ) -> None:
        # Cache the config sections used by the rollout hot path.
        self.config = config
        self.policy = policy
        self.environment = environment
        self.experiment_config = config["experiment"]
        self.model_config = config["model"]
        self.path_config = config["paths"]
        self.runtime_config = config["runtime"]
        self.simulation_config = config["simulation"]
        self.action_config = config["action"]
        self.diagnostic_config = config["collapse_diagnostics"]
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

    # Ensure one CSV file exists with its stable header.
    def _ensure_csv_header(
        self,
        path: Path,
        fieldnames: Sequence[str],
    ) -> None:
        # Create an empty schema file even when no rows are eventually written.
        if path.exists() and path.stat().st_size > 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    # Append one row using a stable CSV header and immediate flush.
    def _append_csv(
        self,
        path: Path,
        fieldnames: Sequence[str],
        row: Mapping[str, Any],
    ) -> None:
        # Retain only declared fields so schema drift cannot corrupt the CSV.
        self._ensure_csv_header(path, fieldnames)
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            if bool(self.logging_config["flush_after_each_rollout"]):
                handle.flush()
                os.fsync(handle.fileno())

    # Read rollout rows belonging only to the current run.
    def _read_rollout_rows(self) -> List[Dict[str, str]]:
        # Filter by run_id so one output directory can retain earlier experiments.
        if not self.rollout_results_path.is_file():
            return []
        with self.rollout_results_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        run_id = str(self.experiment_config["run_id"])
        return [row for row in rows if row.get("run_id") == run_id]

    # Read inference events belonging only to the current run.
    def _read_inference_events(self) -> List[Dict[str, Any]]:
        # Parse JSONL lazily and ignore blank lines left by interrupted writes.
        if not self.inference_events_path.is_file():
            return []
        run_id = str(self.experiment_config["run_id"])
        events: List[Dict[str, Any]] = []
        with self.inference_events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("run_id") == run_id:
                    events.append(event)
        return events

    # Write one inference event immediately for crash-safe records.
    def _write_inference_event(self, event: Mapping[str, Any]) -> None:
        # Flush each model call because a formal run can last many hours.
        if self.inference_handle is None:
            raise RuntimeError("Inference event file is not open.")
        self.inference_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.inference_handle.flush()

    # Convert a complete raw action chunk to the LIBERO control convention.
    def _postprocess_action_chunk(self, action_chunk: np.ndarray) -> np.ndarray:
        # Validate the full chunk before executing any action from it.
        processed = np.asarray(action_chunk, dtype=np.float32).copy()
        expected_shape = (
            int(self.action_config["action_chunk_size"]),
            int(self.action_config["action_dim"]),
        )
        if processed.shape != expected_shape:
            raise ValueError(
                f"Expected action chunk shape {expected_shape}, got {processed.shape}."
            )
        if bool(self.action_config["reject_nan_or_inf"]) and not np.isfinite(
            processed
        ).all():
            raise ValueError("Action chunk contains NaN or Inf.")

        if bool(self.action_config["convert_gripper_to_libero_convention"]):
            processed[:, -1] = 2.0 * processed[:, -1] - 1.0
            if bool(self.action_config["binarize_gripper"]):
                processed[:, -1] = np.sign(processed[:, -1])
            processed[:, -1] *= -1.0

        if bool(self.action_config["clip_actions"]):
            lower = np.asarray(
                self.action_config["action_min"],
                dtype=np.float32,
            )
            upper = np.asarray(
                self.action_config["action_max"],
                dtype=np.float32,
            )
            processed = np.clip(processed, lower, upper)
        return processed

    # Classify one model-side exception into the stable failure vocabulary.
    @staticmethod
    def _classify_model_error(error: BaseException) -> str:
        # Prefer shape and finite-value categories before the generic model error.
        message = str(error).lower()
        if "shape" in message:
            return "invalid_action_shape"
        if "nan" in message or "inf" in message:
            return "nan_or_inf_action"
        if "decode" in message:
            return "action_decode_error"
        return "model_inference_error"

    # Open one rollout video writer when video recording is enabled.
    def _create_video_writer(
        self,
        video_path: Path,
    ) -> Optional[cv2.VideoWriter]:
        # Match DF-05-01 resolution, side-by-side layout, codec, and FPS.
        if not bool(self.video_config["enabled"]):
            return None
        resolution = int(self.config["input"]["environment_render_resolution"])
        width = resolution * (
            2 if bool(self.video_config["include_wrist_view"]) else 1
        )
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

    # Write one annotated primary-and-wrist frame.
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
        # Keep video overlays identical across model variants where possible.
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
        frame = cv2.cvtColor(
            np.ascontiguousarray(frame_rgb),
            cv2.COLOR_RGB2BGR,
        )

        lines: List[str] = []
        if bool(self.video_config["show_model_name"]):
            lines.append(str(self.model_config["model_variant"]))
        if bool(self.video_config["show_task_instruction"]):
            lines.append(str(task_instruction)[:90])
        status_parts: List[str] = []
        if bool(self.video_config["show_trial_index"]):
            status_parts.append(f"trial={trial_index}")
        if bool(self.video_config["show_step_number"]):
            status_parts.append(f"step={step}")
        if (
            bool(self.video_config["show_inference_latency"])
            and inference_latency_ms is not None
        ):
            status_parts.append(f"inference={inference_latency_ms:.1f}ms")
        if bool(self.video_config["show_success_status"]):
            status_parts.append(f"success={success}")
        if status_parts:
            lines.append(" | ".join(status_parts))

        overlay_height = max(28, 24 * len(lines))
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], overlay_height),
            (0, 0, 0),
            -1,
        )
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

    # Return a weighted action control frequency for chunk-level events.
    @staticmethod
    def _control_frequency_hz(
        executed_actions: int,
        elapsed_ms: float,
    ) -> Optional[float]:
        # Divide executed actions by total closed-loop wall time.
        if executed_actions <= 0 or elapsed_ms <= 0.0:
            return None
        return float(executed_actions * 1000.0 / elapsed_ms)

    # Log one collapse warning without changing or rejecting any action.
    def _warn_collapse(
        self,
        rollout_id: str,
        message: str,
    ) -> None:
        # Explicitly state diagnostic-only behavior in formal logs.
        logging.warning(
            "%s collapse_diagnostic=%s diagnostic_only=True",
            rollout_id,
            message,
        )

    # Execute one complete rollout and return its result plus optional failure row.
    def _run_rollout(
        self,
        task_info: Mapping[str, Any],
        trial_index: int,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], bool]:
        # Initialize all counters before touching simulator or video resources.
        task_index = int(task_info["task_index"])
        rollout_id = f"task_{task_index:02d}_trial_{trial_index:03d}"
        rollout_seed = (
            int(self.experiment_config["seed"])
            + task_index * 1000
            + trial_index
        )
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
        zero_arm_count = 0
        full_noop_count = 0
        repeated_action_count = 0
        repeated_chunk_count = 0
        consecutive_zero_actions = 0
        max_consecutive_zero_actions = 0
        consecutive_repeated_chunks = 0
        max_consecutive_repeated_chunks = 0
        collapse_warning_count = 0

        inference_latencies: List[float] = []
        executed_chunk_inference_latencies: List[float] = []
        environment_step_latencies: List[float] = []
        end_to_end_latencies: List[float] = []
        last_action: Optional[np.ndarray] = None
        previous_action: Optional[np.ndarray] = None
        previous_raw_chunk: Optional[np.ndarray] = None

        extension = str(self.video_config["file_extension"])
        video_path = self.video_directory / f"{rollout_id}{extension}"
        writer: Optional[cv2.VideoWriter] = None
        torch.cuda.reset_peak_memory_stats(self.policy.device)
        observation = self.environment.reset(
            task_info["initial_states"][trial_index]
        )

        try:
            writer = self._create_video_writer(video_path)

            # Let movable objects settle before the first model query.
            for wait_step in range(
                int(self.simulation_config["initial_wait_steps"])
            ):
                observation, _, _, _ = self.environment.step(
                    self.environment.dummy_action()
                )
                observation_data = self.environment.extract_observation(
                    observation
                )
                self._write_video_frame(
                    writer,
                    observation_data,
                    str(task_info["task_instruction"]),
                    trial_index,
                    wait_step
                    - int(self.simulation_config["initial_wait_steps"]),
                    None,
                    False,
                )

            max_steps = int(self.simulation_config["max_steps_per_episode"])
            execute_per_inference = int(
                self.action_config["actions_executed_per_inference"]
            )
            while episode_steps < max_steps:
                # Query once, then execute the returned actions open-loop in order.
                chunk_cycle_started = time.perf_counter()
                chunk_start_step = episode_steps
                observation_data = self.environment.extract_observation(
                    observation
                )
                inference_calls += 1

                try:
                    raw_chunk, inference_info = self.policy.predict(
                        primary_image=observation_data["primary_image"],
                        wrist_image=observation_data["wrist_image"],
                        proprio=observation_data["proprio"],
                        instruction=str(task_info["task_instruction"]),
                    )
                    inference_latency_ms = float(
                        inference_info["inference_latency_ms"]
                    )
                    processed_chunk = self._postprocess_action_chunk(raw_chunk)
                    inference_latencies.append(inference_latency_ms)
                    actions_generated += int(
                        inference_info["returned_chunk_size"]
                    )
                except Exception as error:
                    # Record model and action validation errors before terminating.
                    invalid_action_count += 1
                    termination_reason = self._classify_model_error(error)
                    failure_detail = f"{type(error).__name__}: {error}"
                    self._write_inference_event(
                        {
                            "run_id": self.experiment_config["run_id"],
                            "rollout_id": rollout_id,
                            "task_index": task_index,
                            "inference_index": inference_calls,
                            "environment_step": chunk_start_step,
                            "returned_action_shape": None,
                            "returned_chunk_size": 0,
                            "executed_chunk_size": 0,
                            "inference_latency_ms": None,
                            "inference_path": "continuous_l1_action_head",
                            "discrete_action_token_decoding_used": False,
                            "amortized_inference_ms_per_action": None,
                            "environment_step_latency_ms": None,
                            "environment_step_latency_total_ms": None,
                            "end_to_end_latency_ms": None,
                            "amortized_end_to_end_ms_per_action": None,
                            "control_frequency_hz": None,
                            "invalid_action": True,
                            "nan_or_inf_action": (
                                "nan" in str(error).lower()
                                or "inf" in str(error).lower()
                            ),
                            "zero_equivalent_arm_mask": [],
                            "zero_equivalent_full_noop_mask_at_query": [],
                            "zero_equivalent_full_noop_mask": [],
                            "repeated_chunk": False,
                            "normalized_proprio": None,
                            "predicted_action_chunk": None,
                            "executed_action_chunk": [],
                            "error_message": failure_detail,
                            "timestamp": utc_now(),
                        }
                    )
                    break

                repeated_chunk = bool(
                    previous_raw_chunk is not None
                    and np.allclose(
                        raw_chunk,
                        previous_raw_chunk,
                        atol=float(
                            self.diagnostic_config["repeated_chunk_atol"]
                        ),
                        rtol=0.0,
                    )
                )
                if repeated_chunk:
                    repeated_chunk_count += 1
                    consecutive_repeated_chunks += 1
                else:
                    consecutive_repeated_chunks = 0
                max_consecutive_repeated_chunks = max(
                    max_consecutive_repeated_chunks,
                    consecutive_repeated_chunks,
                )
                previous_raw_chunk = raw_chunk.copy()
                repeated_chunk_warning = int(
                    self.diagnostic_config["warn_consecutive_repeated_chunks"]
                )
                if (
                    repeated_chunk_warning > 0
                    and consecutive_repeated_chunks == repeated_chunk_warning
                ):
                    collapse_warning_count += 1
                    self._warn_collapse(
                        rollout_id,
                        f"consecutive_repeated_chunks={consecutive_repeated_chunks}",
                    )

                remaining_steps = max_steps - episode_steps
                requested_chunk_size = min(
                    execute_per_inference,
                    int(inference_info["returned_chunk_size"]),
                    remaining_steps,
                )
                executed_chunk: List[List[float]] = []
                chunk_step_latencies: List[float] = []
                environment_error = ""
                arm_mask = list(inference_info["zero_equivalent_arm_mask"])
                query_full_noop_mask = list(
                    inference_info[
                        "zero_equivalent_full_noop_mask_at_query"
                    ]
                )
                executed_full_noop_mask: List[bool] = []
                current_observation_data = observation_data

                for action_index in range(requested_chunk_size):
                    # Execute each chunk element, update observation, and check success.
                    action = processed_chunk[action_index]
                    current_proprio = np.asarray(
                        current_observation_data["proprio"],
                        dtype=np.float32,
                    )
                    current_gripper_open = bool(
                        float(current_proprio[6] - current_proprio[7])
                        >= self.policy.gripper_aperture_threshold
                    )
                    commanded_gripper_open = bool(
                        raw_chunk[action_index, 6] >= 0.5
                    )
                    full_noop = bool(arm_mask[action_index]) and (
                        commanded_gripper_open == current_gripper_open
                    )
                    action_step_started = time.perf_counter()
                    try:
                        observation, _, done, _ = self.environment.step(action)
                    except Exception as error:
                        termination_reason = "environment_error"
                        environment_error = f"{type(error).__name__}: {error}"
                        failure_detail = environment_error
                        break
                    environment_step_latency_ms = (
                        time.perf_counter() - action_step_started
                    ) * 1000.0
                    chunk_step_latencies.append(environment_step_latency_ms)
                    environment_step_latencies.append(
                        environment_step_latency_ms
                    )
                    episode_steps += 1
                    actions_executed += 1
                    last_action = action.copy()
                    executed_chunk.append(action.tolist())
                    executed_full_noop_mask.append(full_noop)

                    if bool(arm_mask[action_index]):
                        zero_arm_count += 1
                        consecutive_zero_actions += 1
                    else:
                        consecutive_zero_actions = 0
                    if full_noop:
                        full_noop_count += 1
                    max_consecutive_zero_actions = max(
                        max_consecutive_zero_actions,
                        consecutive_zero_actions,
                    )

                    repeated_action = bool(
                        previous_action is not None
                        and np.allclose(
                            action,
                            previous_action,
                            atol=float(
                                self.diagnostic_config["repeated_action_atol"]
                            ),
                            rtol=0.0,
                        )
                    )
                    if repeated_action:
                        repeated_action_count += 1
                    previous_action = action.copy()

                    zero_warning = int(
                        self.diagnostic_config[
                            "warn_consecutive_zero_equivalent_actions"
                        ]
                    )
                    if (
                        zero_warning > 0
                        and consecutive_zero_actions == zero_warning
                    ):
                        collapse_warning_count += 1
                        self._warn_collapse(
                            rollout_id,
                            "consecutive_zero_equivalent_actions="
                            f"{consecutive_zero_actions}",
                        )

                    if bool(self.logging_config["print_action_shape"]):
                        logging.info(
                            "%s action_shape=%s",
                            rollout_id,
                            action.shape,
                        )
                    if bool(self.logging_config["print_action_values"]):
                        logging.info(
                            "%s action=%s",
                            rollout_id,
                            action.tolist(),
                        )

                    success = bool(done)
                    next_observation_data = self.environment.extract_observation(
                        observation
                    )
                    current_observation_data = next_observation_data
                    self._write_video_frame(
                        writer,
                        next_observation_data,
                        str(task_info["task_instruction"]),
                        trial_index,
                        episode_steps,
                        inference_latency_ms,
                        success,
                    )

                    if bool(self.timing_config["enforce_real_time_control"]):
                        target_period = 1.0 / float(
                            self.timing_config["target_control_frequency_hz"]
                        )
                        remaining = target_period - (
                            time.perf_counter() - action_step_started
                        )
                        if remaining > 0.0:
                            time.sleep(remaining)

                    log_interval = int(
                        self.logging_config["log_every_environment_steps"]
                    )
                    if log_interval > 0 and episode_steps % log_interval == 0:
                        logging.info(
                            "%s step=%d inference_ms=%.3f chunk_index=%d",
                            rollout_id,
                            episode_steps,
                            inference_latency_ms,
                            action_index,
                        )
                    if success and bool(
                        self.simulation_config["stop_on_success"]
                    ):
                        termination_reason = "success"
                        break

                executed_chunk_size = len(executed_chunk)
                chunk_end_to_end_ms = (
                    time.perf_counter() - chunk_cycle_started
                ) * 1000.0
                if executed_chunk_size > 0:
                    amortized_inference = (
                        inference_latency_ms / executed_chunk_size
                    )
                    amortized_end_to_end = (
                        chunk_end_to_end_ms / executed_chunk_size
                    )
                    executed_chunk_inference_latencies.append(
                        inference_latency_ms
                    )
                    end_to_end_latencies.append(chunk_end_to_end_ms)
                else:
                    amortized_inference = None
                    amortized_end_to_end = None

                event = {
                    "run_id": self.experiment_config["run_id"],
                    "rollout_id": rollout_id,
                    "task_index": task_index,
                    "inference_index": inference_calls,
                    "environment_step": chunk_start_step,
                    "returned_action_shape": inference_info[
                        "returned_action_shape"
                    ],
                    "returned_chunk_size": inference_info[
                        "returned_chunk_size"
                    ],
                    "executed_chunk_size": executed_chunk_size,
                    "inference_latency_ms": inference_latency_ms,
                    "inference_path": inference_info["inference_path"],
                    "discrete_action_token_decoding_used": inference_info[
                        "discrete_action_token_decoding_used"
                    ],
                    "amortized_inference_ms_per_action": (
                        amortized_inference
                    ),
                    "environment_step_latency_ms": optional_mean(
                        chunk_step_latencies
                    ),
                    "environment_step_latency_total_ms": (
                        float(sum(chunk_step_latencies))
                        if chunk_step_latencies
                        else None
                    ),
                    "end_to_end_latency_ms": (
                        chunk_end_to_end_ms
                        if executed_chunk_size > 0
                        else None
                    ),
                    "amortized_end_to_end_ms_per_action": (
                        amortized_end_to_end
                    ),
                    "control_frequency_hz": self._control_frequency_hz(
                        executed_chunk_size,
                        chunk_end_to_end_ms,
                    ),
                    "invalid_action": False,
                    "nan_or_inf_action": False,
                    "zero_equivalent_arm_mask": arm_mask,
                    "zero_equivalent_full_noop_mask_at_query": (
                        query_full_noop_mask
                    ),
                    "zero_equivalent_full_noop_mask": (
                        executed_full_noop_mask
                    ),
                    "repeated_chunk": repeated_chunk,
                    "normalized_proprio": inference_info[
                        "normalized_proprio"
                    ],
                    "predicted_action_chunk": (
                        raw_chunk.tolist()
                        if bool(
                            self.diagnostic_config[
                                "record_predicted_action_chunks"
                            ]
                        )
                        else None
                    ),
                    "executed_action_chunk": (
                        executed_chunk
                        if bool(
                            self.diagnostic_config[
                                "record_executed_action_chunks"
                            ]
                        )
                        else None
                    ),
                    "error_message": environment_error,
                    "timestamp": utc_now(),
                }
                self._write_inference_event(event)

                if success or environment_error:
                    break

            if not success and not termination_reason:
                timeout = episode_steps >= int(
                    self.simulation_config["max_steps_per_episode"]
                )
                termination_reason = (
                    "timeout" if timeout else "unknown_failure"
                )
        except KeyboardInterrupt:
            # Preserve an interrupted rollout row before propagating the interrupt.
            interrupted = True
            termination_reason = "interrupted"
            failure_detail = "KeyboardInterrupt"
        except Exception as error:
            # Record unexpected simulator, video, or bookkeeping failures.
            termination_reason = "environment_error"
            failure_detail = f"{type(error).__name__}: {error}"
        finally:
            # Always close the video writer so the MP4 container is finalized.
            if writer is not None:
                writer.release()

        keep_video = bool(self.video_config["enabled"]) and (
            success
            and bool(self.video_config["save_success_videos"])
            or not success
            and bool(self.video_config["save_failure_videos"])
        )
        if not keep_video and video_path.exists():
            video_path.unlink()

        if not success and not failure_detail:
            # Attach non-intervening collapse metrics to ordinary timeout failures.
            failure_detail = "collapse_diagnostics=" + json.dumps(
                {
                    "zero_equivalent_arm_action_count": zero_arm_count,
                    "zero_equivalent_full_noop_count": full_noop_count,
                    "repeated_action_count": repeated_action_count,
                    "repeated_chunk_count": repeated_chunk_count,
                    "max_consecutive_zero_equivalent_actions": (
                        max_consecutive_zero_actions
                    ),
                    "max_consecutive_repeated_chunks": (
                        max_consecutive_repeated_chunks
                    ),
                },
                separators=(",", ":"),
            )

        episode_time_seconds = time.perf_counter() - episode_started
        total_closed_loop_ms = float(sum(end_to_end_latencies))
        control_frequency_hz = self._control_frequency_hz(
            actions_executed,
            total_closed_loop_ms,
        )
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
            "p95_inference_latency_ms": optional_percentile(
                inference_latencies,
                95,
            ),
            "avg_amortized_inference_ms_per_action": (
                sum(executed_chunk_inference_latencies) / actions_executed
                if actions_executed
                else None
            ),
            "avg_environment_step_latency_ms": optional_mean(
                environment_step_latencies
            ),
            "p95_environment_step_latency_ms": optional_percentile(
                environment_step_latencies,
                95,
            ),
            "avg_end_to_end_latency_ms": optional_mean(
                end_to_end_latencies
            ),
            "p95_end_to_end_latency_ms": optional_percentile(
                end_to_end_latencies,
                95,
            ),
            "avg_amortized_end_to_end_ms_per_action": (
                total_closed_loop_ms / actions_executed
                if actions_executed
                else None
            ),
            "control_frequency_hz": control_frequency_hz,
            "peak_vram_allocated_gb": torch.cuda.max_memory_allocated(
                self.policy.device
            )
            / 1024**3,
            "peak_vram_reserved_gb": torch.cuda.max_memory_reserved(
                self.policy.device
            )
            / 1024**3,
            "invalid_action_count": invalid_action_count,
            "zero_equivalent_arm_action_count": zero_arm_count,
            "zero_equivalent_full_noop_count": full_noop_count,
            "repeated_action_count": repeated_action_count,
            "repeated_chunk_count": repeated_chunk_count,
            "max_consecutive_zero_equivalent_actions": (
                max_consecutive_zero_actions
            ),
            "max_consecutive_repeated_chunks": (
                max_consecutive_repeated_chunks
            ),
            "collapse_warning_count": collapse_warning_count,
            "automatic_failure_reason": (
                "" if success else termination_reason
            ),
            "failure_detail": failure_detail,
            "manual_failure_reason": "",
            "manual_failure_note": "",
            "video_path": str(video_path) if keep_video else "",
            "started_at": started_at,
            "finished_at": utc_now(),
        }

        failure_row = None
        if not success:
            # Mirror the most useful automatic and diagnostic fields for review.
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
                "zero_equivalent_arm_action_count": zero_arm_count,
                "zero_equivalent_full_noop_count": full_noop_count,
                "repeated_action_count": repeated_action_count,
                "repeated_chunk_count": repeated_chunk_count,
                "last_action": (
                    json.dumps(last_action.tolist())
                    if last_action is not None
                    else ""
                ),
                "video_path": str(video_path) if keep_video else "",
                "timestamp": utc_now(),
            }
        return result, failure_row, interrupted

    # Return a package version without failing when metadata is unavailable.
    @staticmethod
    def _package_version(package_name: str) -> Optional[str]:
        # Preserve null for editable or unavailable package metadata.
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    # Return one Git value from the project repository.
    def _git_value(self, arguments: Sequence[str]) -> Optional[str]:
        # Query Git without mutating the working tree.
        result = subprocess.run(
            ["git", "-C", self.path_config["project_root"], *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    # Build the complete reproducibility manifest for the current run.
    def _build_runtime_manifest(
        self,
        finished_at: Optional[str],
    ) -> Dict[str, Any]:
        # Record both training provenance and actual simulation runtime identity.
        task_ids = [
            int(value) for value in self.simulation_config["task_ids"]
        ]
        training_model = self.policy.training_runtime.get("model", {})
        training_dataset = self.policy.training_runtime.get("dataset", {})
        training_parameters = self.policy.training_runtime.get(
            "parameters",
            {},
        )
        gpu_properties = torch.cuda.get_device_properties(self.policy.device)
        return {
            "task_id": self.experiment_config["task_id"],
            "run_id": self.experiment_config["run_id"],
            "model_variant": self.model_config["model_variant"],
            "model_family": self.model_config["model_family"],
            "policy_type": self.model_config["policy_type"],
            "training_run_id": self.model_config["training_run_id"],
            "checkpoint_step": self.model_config["checkpoint_step"],
            "best_val_loss": self.model_config["best_val_loss"],
            "training_objective": self.model_config["training_objective"],
            "action_representation": self.model_config[
                "action_representation"
            ],
            "use_l1_regression": self.model_config["use_l1_regression"],
            "use_action_head": self.model_config["use_action_head"],
            "use_diffusion": self.model_config["use_diffusion"],
            "use_film": self.model_config["use_film"],
            "strict_continuous_action_path": self.model_config[
                "strict_continuous_action_path"
            ],
            "continuous_action_head_required": True,
            "discrete_action_token_decoding_used": False,
            "discrete_action_fallback_allowed": False,
            "collapse_action_intervention_enabled": False,
            "action_dim": self.action_config["action_dim"],
            "action_chunk_size": self.action_config["action_chunk_size"],
            "actions_executed_per_inference": self.action_config[
                "actions_executed_per_inference"
            ],
            "replan_frequency": self.action_config[
                "actions_executed_per_inference"
            ],
            "suite_name": self.simulation_config["task_suite_name"],
            "selected_task_ids": task_ids,
            "num_tasks": len(task_ids),
            "num_trials_per_task": self.simulation_config[
                "num_trials_per_task"
            ],
            "max_episode_steps": self.simulation_config[
                "max_steps_per_episode"
            ],
            "initial_wait_steps": self.simulation_config[
                "initial_wait_steps"
            ],
            "seed": self.experiment_config["seed"],
            "environment_seed": self.simulation_config["environment_seed"],
            "use_wrist_image": self.config["input"]["use_wrist_image"],
            "use_proprio": self.config["input"]["use_proprio"],
            "num_images_in_input": self.config["input"][
                "num_images_in_input"
            ],
            "proprio_dim": self.config["input"]["proprio_dim"],
            "torch_dtype": self.model_config["torch_dtype"],
            "device": self.model_config["device"],
            "attn_implementation": self.model_config[
                "attn_implementation"
            ],
            "simulation_gpu_name": torch.cuda.get_device_name(
                self.policy.device
            ),
            "gpu_name": torch.cuda.get_device_name(self.policy.device),
            "simulation_gpu_total_memory_gb": (
                gpu_properties.total_memory / 1024**3
            ),
            "expected_gpu_name": self.runtime_config["expected_gpu_name"],
            "training_gpu_name": training_model.get("gpu_name"),
            "training_dataset": training_dataset,
            "training_parameters": training_parameters,
            "base_model_path": self.path_config["base_model_path"],
            "adapter_path": self.path_config["adapter_path"],
            "processor_path": self.path_config["processor_path"],
            "action_head_path": self.path_config["action_head_path"],
            "proprio_projector_path": self.path_config[
                "proprio_projector_path"
            ],
            "dataset_statistics_path": self.path_config[
                "dataset_statistics_path"
            ],
            "training_config_path": self.path_config[
                "training_config_path"
            ],
            "training_summary_path": self.path_config[
                "training_summary_path"
            ],
            "openvla_source_root": self.path_config[
                "openvla_source_root"
            ],
            "artifact_sha256": self.policy.artifact_sha256,
            "zero_equivalent_action_definition": {
                "description": (
                    "Continuous arm output lies inside each dimension's "
                    "physical interval corresponding to the OpenVLA zero bin."
                ),
                "bounds": self.policy.zero_equivalent_bounds,
                "gripper_aperture_threshold": (
                    self.policy.gripper_aperture_threshold
                ),
                "diagnostic_only": True,
            },
            "collapse_diagnostics": self.diagnostic_config,
            "git_branch": self._git_value(["branch", "--show-current"]),
            "git_commit": self._git_value(["rev-parse", "HEAD"]),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
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

    # Return task events and common latency aggregates.
    def _event_aggregates(
        self,
        events: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        # Compute weighted control frequency from actual executed chunk sizes.
        inference_values = [
            float(event["inference_latency_ms"])
            for event in events
            if event.get("inference_latency_ms") is not None
        ]
        environment_values = [
            float(event["environment_step_latency_ms"])
            for event in events
            if event.get("environment_step_latency_ms") is not None
        ]
        environment_total_values = [
            float(event["environment_step_latency_total_ms"])
            for event in events
            if event.get("environment_step_latency_total_ms") is not None
        ]
        end_to_end_values = [
            float(event["end_to_end_latency_ms"])
            for event in events
            if event.get("end_to_end_latency_ms") is not None
        ]
        executed_inference_values = [
            float(event["inference_latency_ms"])
            for event in events
            if int(event.get("executed_chunk_size", 0)) > 0
            and event.get("inference_latency_ms") is not None
        ]
        executed_actions = sum(
            int(event.get("executed_chunk_size", 0)) for event in events
        )
        total_end_to_end_ms = sum(end_to_end_values)
        return {
            "avg_inference_latency_ms": optional_mean(inference_values),
            "p95_inference_latency_ms": optional_percentile(
                inference_values,
                95,
            ),
            "avg_amortized_inference_ms_per_action": (
                sum(executed_inference_values) / executed_actions
                if executed_actions
                else None
            ),
            "avg_environment_step_latency_ms": (
                sum(environment_total_values) / executed_actions
                if executed_actions
                else None
            ),
            "p95_environment_step_latency_ms": optional_percentile(
                environment_values,
                95,
            ),
            "avg_end_to_end_latency_ms": optional_mean(end_to_end_values),
            "p95_end_to_end_latency_ms": optional_percentile(
                end_to_end_values,
                95,
            ),
            "avg_amortized_end_to_end_ms_per_action": (
                total_end_to_end_ms / executed_actions
                if executed_actions
                else None
            ),
            "avg_control_frequency_hz": self._control_frequency_hz(
                executed_actions,
                total_end_to_end_ms,
            ),
        }

    # Write per-task and overall summaries from raw CSV and JSONL records.
    def _generate_summaries(
        self,
        requested_completed: bool,
    ) -> Dict[str, Any]:
        # Recompute summaries from raw records so resumed runs remain correct.
        rows = self._read_rollout_rows()
        events = self._read_inference_events()
        task_ids = [
            int(value) for value in self.simulation_config["task_ids"]
        ]
        task_summaries: List[Dict[str, Any]] = []

        for task_index in task_ids:
            task_rows = [
                row
                for row in rows
                if int(row["task_index"]) == task_index
            ]
            task_rollout_ids = {row["rollout_id"] for row in task_rows}
            task_events = [
                event
                for event in events
                if event.get("rollout_id") in task_rollout_ids
            ]
            success_rows = [
                row
                for row in task_rows
                if row["success"].lower() == "true"
            ]
            timeout_rows = [
                row
                for row in task_rows
                if row["timeout"].lower() == "true"
            ]
            total_actions = sum(
                csv_int(row.get("actions_executed")) for row in task_rows
            )
            zero_actions = sum(
                csv_int(row.get("zero_equivalent_arm_action_count"))
                for row in task_rows
            )
            full_noops = sum(
                csv_int(row.get("zero_equivalent_full_noop_count"))
                for row in task_rows
            )
            latency = self._event_aggregates(task_events)
            task_summaries.append(
                {
                    "task_index": task_index,
                    "task_name": (
                        task_rows[0]["task_name"] if task_rows else ""
                    ),
                    "total_rollouts": len(task_rows),
                    "success_count": len(success_rows),
                    "failure_count": len(task_rows) - len(success_rows),
                    "success_rate": (
                        len(success_rows) / len(task_rows)
                        if task_rows
                        else None
                    ),
                    "timeout_count": len(timeout_rows),
                    "timeout_rate": (
                        len(timeout_rows) / len(task_rows)
                        if task_rows
                        else None
                    ),
                    "avg_success_steps": optional_mean(
                        [
                            float(row["episode_steps"])
                            for row in success_rows
                        ]
                    ),
                    "avg_all_steps": optional_mean(
                        [
                            float(row["episode_steps"])
                            for row in task_rows
                        ]
                    ),
                    "avg_episode_time_seconds": optional_mean(
                        [
                            float(row["episode_time_seconds"])
                            for row in task_rows
                        ]
                    ),
                    **latency,
                    "peak_vram_allocated_gb": max(
                        [
                            float(row["peak_vram_allocated_gb"])
                            for row in task_rows
                        ],
                        default=None,
                    ),
                    "peak_vram_reserved_gb": max(
                        [
                            float(row["peak_vram_reserved_gb"])
                            for row in task_rows
                        ],
                        default=None,
                    ),
                    "zero_equivalent_arm_action_count": zero_actions,
                    "zero_equivalent_arm_action_rate": (
                        zero_actions / total_actions
                        if total_actions
                        else None
                    ),
                    "zero_equivalent_full_noop_count": full_noops,
                    "zero_equivalent_full_noop_rate": (
                        full_noops / total_actions
                        if total_actions
                        else None
                    ),
                    "repeated_action_count": sum(
                        csv_int(row.get("repeated_action_count"))
                        for row in task_rows
                    ),
                    "repeated_chunk_count": sum(
                        csv_int(row.get("repeated_chunk_count"))
                        for row in task_rows
                    ),
                    "collapse_warning_count": sum(
                        csv_int(row.get("collapse_warning_count"))
                        for row in task_rows
                    ),
                }
            )

        self.task_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.task_summary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            fieldnames = (
                list(task_summaries[0].keys()) if task_summaries else []
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(task_summaries)

        success_rows = [
            row for row in rows if row["success"].lower() == "true"
        ]
        timeout_rows = [
            row for row in rows if row["timeout"].lower() == "true"
        ]
        expected_rollouts = len(task_ids) * int(
            self.simulation_config["num_trials_per_task"]
        )
        completed = requested_completed and len(rows) == expected_rollouts
        task_success_rates = [
            float(task["success_rate"])
            for task in task_summaries
            if task["success_rate"] is not None
        ]
        total_actions = sum(
            csv_int(row.get("actions_executed")) for row in rows
        )
        zero_actions = sum(
            csv_int(row.get("zero_equivalent_arm_action_count"))
            for row in rows
        )
        full_noops = sum(
            csv_int(row.get("zero_equivalent_full_noop_count"))
            for row in rows
        )
        latency = self._event_aggregates(events)
        summary = {
            "task_id": self.experiment_config["task_id"],
            "run_id": self.experiment_config["run_id"],
            "model_variant": self.model_config["model_variant"],
            "checkpoint_step": self.model_config["checkpoint_step"],
            "total_tasks": len(task_ids),
            "total_rollouts": len(rows),
            "successful_rollouts": len(success_rows),
            "failed_rollouts": len(rows) - len(success_rows),
            "overall_success_rate": (
                len(success_rows) / len(rows) if rows else None
            ),
            "macro_average_task_success_rate": optional_mean(
                task_success_rates
            ),
            "timeout_count": len(timeout_rows),
            "overall_timeout_rate": (
                len(timeout_rows) / len(rows) if rows else None
            ),
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
            "total_actions_generated": sum(
                csv_int(row.get("actions_generated")) for row in rows
            ),
            "total_actions_executed": total_actions,
            **latency,
            "peak_vram_allocated_gb": max(
                [
                    float(row["peak_vram_allocated_gb"])
                    for row in rows
                ],
                default=None,
            ),
            "peak_vram_reserved_gb": max(
                [
                    float(row["peak_vram_reserved_gb"])
                    for row in rows
                ],
                default=None,
            ),
            "zero_equivalent_arm_action_count": zero_actions,
            "zero_equivalent_arm_action_rate": (
                zero_actions / total_actions if total_actions else None
            ),
            "zero_equivalent_full_noop_count": full_noops,
            "zero_equivalent_full_noop_rate": (
                full_noops / total_actions if total_actions else None
            ),
            "repeated_action_count": sum(
                csv_int(row.get("repeated_action_count")) for row in rows
            ),
            "repeated_chunk_count": sum(
                csv_int(row.get("repeated_chunk_count")) for row in rows
            ),
            "collapse_warning_count": sum(
                csv_int(row.get("collapse_warning_count")) for row in rows
            ),
            "rollouts_with_collapse_warning": sum(
                csv_int(row.get("collapse_warning_count")) > 0
                for row in rows
            ),
            "collapse_diagnostics_are_interventions": False,
            "inference_path": "continuous_l1_action_head",
            "discrete_action_token_decoding_used": False,
            "collapse_action_intervention_enabled": False,
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

    # Run every configured task and trial with crash-safe resume support.
    def run(self) -> Dict[str, Any]:
        # Write start-state provenance before the first environment is created.
        if bool(self.recording_config["save_runtime_manifest"]):
            write_json(
                self.runtime_manifest_path,
                self._build_runtime_manifest(finished_at=None),
            )
        self._ensure_csv_header(
            self.rollout_results_path,
            ROLLOUT_FIELDS,
        )
        self._ensure_csv_header(
            self.failure_cases_path,
            FAILURE_FIELDS,
        )

        existing_rows = self._read_rollout_rows()
        completed_rollout_ids = {
            row["rollout_id"] for row in existing_rows
        }
        requested_completed = False
        caught_error: Optional[BaseException] = None
        self.inference_events_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            # Keep the JSONL stream open and line-buffered throughout the run.
            self.inference_handle = self.inference_events_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            for task_index in [
                int(value) for value in self.simulation_config["task_ids"]
            ]:
                task_info = self.environment.prepare_task(task_index)
                trials = int(
                    self.simulation_config["num_trials_per_task"]
                )
                if len(task_info["initial_states"]) < trials:
                    raise ValueError(
                        f"Task {task_index} has "
                        f"{len(task_info['initial_states'])} initial states, "
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
                    # Skip only rows already completed for this exact run_id.
                    rollout_id = (
                        f"task_{task_index:02d}_trial_{trial_index:03d}"
                    )
                    if (
                        bool(
                            self.simulation_config[
                                "resume_from_existing_results"
                            ]
                        )
                        and bool(
                            self.simulation_config[
                                "skip_completed_rollouts"
                            ]
                        )
                        and rollout_id in completed_rollout_ids
                    ):
                        logging.info(
                            "Skipping completed rollout=%s",
                            rollout_id,
                        )
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
            # Delay re-raising until raw files and summaries are finalized.
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
