# Create LIBERO environments and expose the exact DF-04-03 observation inputs.

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# Manage one LIBERO task environment and its deterministic initial states.
class LiberoEnvironment:
    # Initialize the configured LIBERO benchmark suite.
    def __init__(self, config: Mapping[str, Any]) -> None:
        # Retain only the sections used by environment construction and observations.
        self.config = config
        self.input_config = config["input"]
        self.simulation_config = config["simulation"]
        self.task_suite_name = str(self.simulation_config["task_suite_name"])
        benchmark_dict = benchmark.get_benchmark_dict()
        if self.task_suite_name not in benchmark_dict:
            raise KeyError(f"Unknown LIBERO task suite: {self.task_suite_name!r}.")
        self.task_suite = benchmark_dict[self.task_suite_name]()
        self.env: Optional[OffScreenRenderEnv] = None
        self.current_task: Any = None

    # Load official LIBERO initial states with PyTorch 2.6+ compatibility.
    def _load_initial_states(self, task: Any) -> torch.Tensor:
        # Use the benchmark-owned state file instead of generating new resets.
        path = (
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO initial-state file not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    # Convert one xyzw quaternion to the axis-angle representation used in training.
    @staticmethod
    def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
        # Match the official LIBERO evaluation conversion used by OpenVLA-OFT.
        quat = np.asarray(quaternion, dtype=np.float64).copy()
        if quat.shape != (4,):
            raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}.")
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
        if math.isclose(denominator, 0.0):
            return np.zeros(3, dtype=np.float64)
        return quat[:3] * (2.0 * math.acos(quat[3])) / denominator

    # Create the environment for one task and return its reproducible metadata.
    def prepare_task(self, task_index: int) -> Dict[str, Any]:
        # Recreate the off-screen renderer between tasks to release MuJoCo state.
        if task_index < 0 or task_index >= int(self.task_suite.n_tasks):
            raise IndexError(f"LIBERO task index is out of range: {task_index}.")
        self.close()

        task = self.task_suite.get_task(task_index)
        bddl_path = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        resolution = int(self.input_config["environment_render_resolution"])
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        self.env.seed(int(self.simulation_config["environment_seed"]))
        self.current_task = task
        return {
            "task_index": task_index,
            "task_name": Path(task.bddl_file).stem,
            "task_instruction": task.language,
            "initial_states": self._load_initial_states(task),
        }

    # Reset the active task to one selected official initial state.
    def reset(self, initial_state: Any) -> Dict[str, Any]:
        # Reset simulator internals before applying the deterministic state vector.
        if self.env is None:
            raise RuntimeError("LIBERO environment has not been prepared.")
        self.env.reset()
        state = np.asarray(initial_state)
        return self.env.set_init_state(state)

    # Execute one seven-dimensional action and return the updated observation.
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Convert to float32 at the environment boundary for consistent control.
        if self.env is None:
            raise RuntimeError("LIBERO environment has not been prepared.")
        return self.env.step(np.asarray(action, dtype=np.float32).tolist())

    # Return the official no-op action used during initial settling.
    def dummy_action(self) -> np.ndarray:
        # Keep the gripper open while all six motion dimensions remain zero.
        return np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            dtype=np.float32,
        )

    # Extract rotated RGB views and the raw eight-dimensional proprio vector.
    def extract_observation(
        self,
        observation: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        # Require every field used by the DF-04-03 two-image proprio contract.
        required_keys = (
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
        missing = [key for key in required_keys if key not in observation]
        if missing:
            raise KeyError("LIBERO observation is missing: " + ", ".join(missing))

        primary = np.asarray(observation["agentview_image"])
        wrist = np.asarray(observation["robot0_eye_in_hand_image"])
        if bool(self.input_config["rotate_primary_image_180"]):
            primary = primary[::-1, ::-1]
        if bool(self.input_config["rotate_wrist_image_180"]):
            wrist = wrist[::-1, ::-1]

        proprio = np.concatenate(
            (
                np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
                self._quat_to_axis_angle(observation["robot0_eef_quat"]).astype(
                    np.float32
                ),
                np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        expected_dim = int(self.input_config["proprio_dim"])
        if proprio.shape != (expected_dim,):
            raise ValueError(
                f"Expected proprio shape {(expected_dim,)}, got {proprio.shape}."
            )
        if not np.isfinite(proprio).all():
            raise ValueError("Raw LIBERO proprio contains NaN or Inf.")

        return {
            "primary_image": np.ascontiguousarray(primary),
            "wrist_image": np.ascontiguousarray(wrist),
            "proprio": np.ascontiguousarray(proprio),
        }

    # Close the active MuJoCo environment and release its renderer.
    def close(self) -> None:
        # Make repeated task creation safe during long formal evaluations.
        if self.env is not None:
            self.env.close()
            self.env = None
            self.current_task = None
