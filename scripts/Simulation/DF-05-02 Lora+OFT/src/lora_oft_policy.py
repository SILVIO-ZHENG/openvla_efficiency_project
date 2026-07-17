# Load the DF-04-03 LoRA+OFT policy and predict continuous 8-step action chunks.

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import tensorflow as tf
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoConfig, AutoModelForVision2Seq, AutoTokenizer

import prismatic.extern.hf.modeling_prismatic as modeling_prismatic
import prismatic.models.action_heads as action_heads_module
import prismatic.vla.constants as vla_constants
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector


# Keep TensorFlow image preprocessing on CPU so it cannot reserve model VRAM.
tf.config.set_visible_devices([], "GPU")


# Resolve one supported torch dtype from a YAML string.
def resolve_torch_dtype(value: str) -> torch.dtype:
    # Keep accepted dtypes explicit so silent precision changes cannot occur.
    normalized = str(value).strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value!r}.")
    return mapping[normalized]


# Compute one SHA256 digest without loading the complete file into memory.
def sha256_file(path: Path) -> str:
    # Stream large adapter weights in fixed-size blocks.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Load and execute the strict continuous LoRA+OFT inference path.
class LoraOFTPolicy:
    # Initialize model artifacts, continuous heads, statistics, and diagnostics.
    def __init__(self, config: Mapping[str, Any]) -> None:
        # Retain resolved config sections used throughout inference.
        self.config = config
        self.model_config = config["model"]
        self.path_config = config["paths"]
        self.input_config = config["input"]
        self.action_config = config["action"]
        self.recording_config = config["recording"]
        self.timing_config = config["timing"]
        self.training_contract = config["training_contract"]

        # Reject any configuration that could re-enable the DF-05-01 token path.
        if not bool(self.model_config["strict_continuous_action_path"]):
            raise RuntimeError("Strict continuous action inference is required.")
        if not bool(self.model_config["forbid_discrete_action_fallback"]):
            raise RuntimeError("Discrete action fallback is forbidden.")

        self.device = torch.device(self.model_config["device"])
        self.dtype = resolve_torch_dtype(self.model_config["torch_dtype"])
        self.unnorm_key = str(self.training_contract["dataset_name"])
        self.action_dim = int(self.action_config["action_dim"])
        self.action_chunk_size = int(self.action_config["action_chunk_size"])
        self.proprio_dim = int(self.input_config["proprio_dim"])

        self._validate_paths()
        self._validate_source_modules()
        self._validate_constants()
        self.training_runtime = self._load_json(
            Path(self.path_config["training_config_path"])
        )
        self.training_summary = self._load_json(
            Path(self.path_config["training_summary_path"])
        )
        self._validate_training_contract()
        self.statistics = self._load_statistics()
        self.processor = self._load_processor()
        self.model, self.action_model = self._load_model()
        self.action_head = self._load_action_head()
        self.proprio_projector = self._load_proprio_projector()
        self.zero_equivalent_bounds = self._build_zero_equivalent_bounds()
        self.gripper_aperture_threshold = (
            self._compute_gripper_aperture_threshold()
        )
        self.artifact_sha256 = self._compute_artifact_hashes()

    # Load one JSON object and reject any non-mapping root.
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        # Read metadata using UTF-8 for reproducible structured records.
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise TypeError(f"Expected a JSON object in {path}.")
        return document

    # Confirm that every artifact required by continuous inference exists.
    def _validate_paths(self) -> None:
        # Include action_head.pt so missing OFT weights can never trigger fallback.
        source_root = Path(self.path_config["openvla_source_root"])
        required_files = {
            "base model config": Path(self.path_config["base_model_path"])
            / "config.json",
            "DF-04-03 modeling source": source_root
            / "prismatic/extern/hf/modeling_prismatic.py",
            "DF-04-03 action head source": source_root
            / "prismatic/models/action_heads.py",
            "adapter config": Path(self.path_config["adapter_path"])
            / "adapter_config.json",
            "adapter weights": Path(self.path_config["adapter_path"])
            / "adapter_model.safetensors",
            "continuous action head": Path(self.path_config["action_head_path"]),
            "proprio projector": Path(
                self.path_config["proprio_projector_path"]
            ),
            "dataset statistics": Path(
                self.path_config["dataset_statistics_path"]
            ),
            "processor config": Path(self.path_config["processor_path"])
            / "preprocessor_config.json",
            "tokenizer config": Path(self.path_config["processor_path"])
            / "tokenizer_config.json",
            "training runtime config": Path(
                self.path_config["training_config_path"]
            ),
            "training summary": Path(self.path_config["training_summary_path"]),
        }
        missing = [name for name, path in required_files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing LoRA+OFT inference artifacts: " + ", ".join(missing)
            )

    # Confirm imported prismatic modules come from the configured DF-04-03 source.
    def _validate_source_modules(self) -> None:
        # Prevent an older DF-04-02 source tree from being imported accidentally.
        source_root = Path(self.path_config["openvla_source_root"]).resolve()
        modules = {
            "modeling_prismatic": modeling_prismatic,
            "action_heads": action_heads_module,
            "vla_constants": vla_constants,
        }
        for name, module in modules.items():
            module_path = Path(str(module.__file__)).resolve()
            if not module_path.is_relative_to(source_root):
                raise RuntimeError(
                    f"{name} was imported from {module_path}, not {source_root}."
                )

    # Confirm source-level action and proprio constants match the saved run.
    def _validate_constants(self) -> None:
        # Reject any silent one-step override inherited from DF-05-01.
        expected = {
            "ACTION_DIM": self.action_dim,
            "NUM_ACTIONS_CHUNK": self.action_chunk_size,
            "PROPRIO_DIM": self.proprio_dim,
        }
        actual = {
            "ACTION_DIM": int(vla_constants.ACTION_DIM),
            "NUM_ACTIONS_CHUNK": int(vla_constants.NUM_ACTIONS_CHUNK),
            "PROPRIO_DIM": int(vla_constants.PROPRIO_DIM),
        }
        if actual != expected:
            raise RuntimeError(
                f"DF-04-03 constant mismatch: expected={expected}, actual={actual}."
            )
        if int(modeling_prismatic.NUM_ACTIONS_CHUNK) != self.action_chunk_size:
            raise RuntimeError("modeling_prismatic.NUM_ACTIONS_CHUNK must be 8.")
        if int(action_heads_module.NUM_ACTIONS_CHUNK) != self.action_chunk_size:
            raise RuntimeError("action_heads.NUM_ACTIONS_CHUNK must be 8.")

    # Validate the saved runtime and summary against the intended formal run.
    def _validate_training_contract(self) -> None:
        # Use produced runtime records as truth instead of trusting filenames.
        dataset = self.training_runtime.get("dataset", {})
        oft = self.training_runtime.get("oft", {})
        lora = self.training_runtime.get("lora_config", {})
        model = self.training_runtime.get("model", {})
        training = self.training_runtime.get("training", {})
        totals = dataset.get("split_totals", {})

        checks = {
            "dataset.name": (
                dataset.get("name"),
                self.training_contract["dataset_name"],
            ),
            "dataset.split_seed": (
                int(dataset.get("split_seed", -1)),
                int(self.training_contract["split_seed"]),
            ),
            "dataset.train_count": (
                int(totals.get("train_count", -1)),
                int(self.training_contract["train_count"]),
            ),
            "dataset.val_count": (
                int(totals.get("val_count", -1)),
                int(self.training_contract["val_count"]),
            ),
            "dataset.test_count": (
                int(totals.get("test_count", -1)),
                int(self.training_contract["test_count"]),
            ),
            "episode_split_sha256": (
                dataset.get("episode_split_sha256"),
                self.training_contract["episode_split_sha256"],
            ),
            "normalization_stats_sha256": (
                dataset.get("normalization_stats_sha256"),
                self.training_contract["normalization_stats_sha256"],
            ),
            "use_l1_regression": (bool(oft.get("use_l1_regression")), True),
            "use_diffusion": (bool(oft.get("use_diffusion")), False),
            "use_film": (bool(oft.get("use_film")), False),
            "num_images_in_input": (
                int(oft.get("num_images_in_input", -1)),
                2,
            ),
            "use_proprio": (bool(oft.get("use_proprio")), True),
            "action_dim": (int(oft.get("action_dim", -1)), self.action_dim),
            "proprio_dim": (
                int(oft.get("proprio_dim", -1)),
                self.proprio_dim,
            ),
            "num_actions_chunk": (
                int(oft.get("num_actions_chunk", -1)),
                self.action_chunk_size,
            ),
            "normalization_type": (
                oft.get("normalization_type"),
                self.training_contract["normalization_type"],
            ),
            "lora_rank": (
                int(lora.get("r", -1)),
                int(self.training_contract["lora_rank"]),
            ),
            "lora_alpha": (
                int(lora.get("lora_alpha", -1)),
                int(self.training_contract["lora_alpha"]),
            ),
            "lora_dropout": (
                float(lora.get("lora_dropout", -1.0)),
                float(self.training_contract["lora_dropout"]),
            ),
            "per_device_batch_size": (
                int(training.get("per_device_batch_size", -1)),
                int(self.training_contract["per_device_batch_size"]),
            ),
            "max_steps": (
                int(training.get("max_steps", -1)),
                int(self.training_contract["max_steps"]),
            ),
            "torch_dtype": (model.get("requested_torch_dtype"), "bfloat16"),
            "training_status": (
                self.training_summary.get("status"),
                "training_finished",
            ),
            "total_training_steps": (
                int(self.training_summary.get("total_training_steps", -1)),
                int(self.training_contract["max_steps"]),
            ),
            "best_checkpoint_step": (
                int(self.training_summary.get("best_checkpoint_step", -1)),
                int(self.model_config["checkpoint_step"]),
            ),
        }
        mismatches = [
            f"{name}: actual={actual!r}, expected={expected!r}"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if mismatches:
            raise RuntimeError(
                "DF-04-03 training contract mismatch: " + "; ".join(mismatches)
            )
        actual_best_loss = float(
            self.training_summary.get("best_val_loss", float("nan"))
        )
        expected_best_loss = float(self.model_config["best_val_loss"])
        if not np.isclose(actual_best_loss, expected_best_loss, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "DF-04-03 best validation loss mismatch: "
                f"actual={actual_best_loss}, expected={expected_best_loss}."
            )

    # Load and validate the train-only normalization statistics bundle.
    def _load_statistics(self) -> Dict[str, Any]:
        # The best adapter stores the compatibility mapping directly by dataset key.
        statistics = self._load_json(
            Path(self.path_config["dataset_statistics_path"])
        )
        if self.unnorm_key not in statistics:
            raise KeyError(f"Missing statistics key: {self.unnorm_key!r}.")
        dataset_stats = statistics[self.unnorm_key]
        action_stats = dataset_stats.get("action", {})
        proprio_stats = dataset_stats.get("proprio", {})

        action_q01 = np.asarray(action_stats.get("q01", []), dtype=np.float64)
        action_q99 = np.asarray(action_stats.get("q99", []), dtype=np.float64)
        proprio_q01 = np.asarray(proprio_stats.get("q01", []), dtype=np.float64)
        proprio_q99 = np.asarray(proprio_stats.get("q99", []), dtype=np.float64)
        action_mask = np.asarray(action_stats.get("mask", []), dtype=bool)
        proprio_mask = np.asarray(proprio_stats.get("mask", []), dtype=bool)

        if action_q01.shape != (self.action_dim,) or action_q99.shape != (
            self.action_dim,
        ):
            raise ValueError("Action q01/q99 statistics must each contain 7 values.")
        if proprio_q01.shape != (self.proprio_dim,) or proprio_q99.shape != (
            self.proprio_dim,
        ):
            raise ValueError("Proprio q01/q99 statistics must each contain 8 values.")
        if action_mask.tolist() != [True] * 6 + [False]:
            raise ValueError("Action mask must normalize arm6 and preserve gripper.")
        if proprio_mask.tolist() != [True] * self.proprio_dim:
            raise ValueError("All eight proprio dimensions must be normalized.")
        if int(dataset_stats.get("num_trajectories", -1)) != int(
            self.training_contract["train_count"]
        ):
            raise ValueError("Statistics must come from all 330 training episodes.")
        if dataset_stats.get("source_split") != "train":
            raise ValueError("Statistics source_split must be 'train'.")
        return statistics

    # Reconstruct the saved processor without requiring processor_config.json.
    def _load_processor(self) -> PrismaticProcessor:
        # Load image and text processors exclusively from the best adapter bundle.
        processor_path = Path(self.path_config["processor_path"])
        image_processor = PrismaticImageProcessor.from_pretrained(
            processor_path,
            local_files_only=bool(self.model_config["local_files_only"]),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            processor_path,
            local_files_only=bool(self.model_config["local_files_only"]),
        )
        return PrismaticProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

    # Load OpenVLA and attach the trained LoRA adapter.
    def _load_model(self) -> Tuple[PeftModel, OpenVLAForActionPrediction]:
        # Register local model classes because trust_remote_code remains disabled.
        AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        AutoModelForVision2Seq.register(
            OpenVLAConfig,
            OpenVLAForActionPrediction,
            exist_ok=True,
        )
        base_model = AutoModelForVision2Seq.from_pretrained(
            self.path_config["base_model_path"],
            attn_implementation=self.model_config["attn_implementation"],
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=bool(self.model_config["trust_remote_code"]),
            local_files_only=bool(self.model_config["local_files_only"]),
        )
        base_model.vision_backbone.set_num_images_in_input(
            int(self.input_config["num_images_in_input"])
        )
        base_model.norm_stats = self.statistics

        model = PeftModel.from_pretrained(
            base_model,
            self.path_config["adapter_path"],
            is_trainable=False,
            local_files_only=bool(self.model_config["local_files_only"]),
        ).to(self.device)
        model.eval()

        action_model = model.get_base_model()
        action_model.norm_stats = self.statistics
        action_model.vision_backbone.set_num_images_in_input(
            int(self.input_config["num_images_in_input"])
        )
        return model, action_model

    # Load one saved component state and remove an optional DDP prefix.
    @staticmethod
    def _load_component_state(path: Path) -> Dict[str, torch.Tensor]:
        # Accept historical DDP checkpoints while retaining strict key validation.
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise TypeError(f"Component state is empty or invalid: {path}")
        return {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state.items()
        }

    # Recreate and strictly load the continuous L1 action head.
    def _load_action_head(self) -> L1RegressionActionHead:
        # A non-null action head is mandatory to prevent discrete-token fallback.
        head = L1RegressionActionHead(
            input_dim=int(self.action_model.llm_dim),
            hidden_dim=int(self.action_model.llm_dim),
            action_dim=self.action_dim,
        )
        state = self._load_component_state(
            Path(self.path_config["action_head_path"])
        )
        head.load_state_dict(state, strict=True)
        head.to(device=self.device, dtype=self.dtype)
        head.eval()
        if not isinstance(head, L1RegressionActionHead):
            raise TypeError("Loaded action head is not L1RegressionActionHead.")
        return head

    # Recreate and strictly load the eight-dimensional proprio projector.
    def _load_proprio_projector(self) -> ProprioProjector:
        # Match the exact llm_dim and proprio_dim used in DF-04-03 training.
        projector = ProprioProjector(
            llm_dim=int(self.action_model.llm_dim),
            proprio_dim=self.proprio_dim,
        )
        state = self._load_component_state(
            Path(self.path_config["proprio_projector_path"])
        )
        projector.load_state_dict(state, strict=True)
        projector.to(device=self.device, dtype=self.dtype)
        projector.eval()
        return projector

    # Compute reproducibility hashes for every loaded model-side artifact.
    def _compute_artifact_hashes(self) -> Dict[str, str]:
        # Make hashing configurable because adapter weights can be large.
        if not bool(self.recording_config["compute_artifact_sha256"]):
            return {}
        paths = {
            "adapter_config": Path(self.path_config["adapter_path"])
            / "adapter_config.json",
            "adapter_weights": Path(self.path_config["adapter_path"])
            / "adapter_model.safetensors",
            "action_head": Path(self.path_config["action_head_path"]),
            "proprio_projector": Path(
                self.path_config["proprio_projector_path"]
            ),
            "dataset_statistics": Path(
                self.path_config["dataset_statistics_path"]
            ),
            "training_config": Path(self.path_config["training_config_path"]),
            "training_summary": Path(self.path_config["training_summary_path"]),
        }
        return {name: sha256_file(path) for name, path in paths.items()}

    # Derive physical intervals equivalent to each arm dimension's zero bin.
    def _build_zero_equivalent_bounds(self) -> Dict[str, Any]:
        # Reuse OpenVLA's own bin boundaries only as a diagnostic reference.
        stats = self.statistics[self.unnorm_key]["action"]
        low = np.asarray(stats["q01"], dtype=np.float64)[:6]
        high = np.asarray(stats["q99"], dtype=np.float64)[:6]
        normalized_zero = 2.0 * (0.0 - low) / (high - low + 1e-8) - 1.0
        boundaries = np.asarray(self.action_model.bins, dtype=np.float64)
        indices = np.searchsorted(
            boundaries,
            normalized_zero,
            side="right",
        ) - 1
        indices = np.clip(indices, 0, len(boundaries) - 2)
        normalized_lower = boundaries[indices]
        normalized_upper = boundaries[indices + 1]
        raw_lower = (
            0.5 * (normalized_lower + 1.0) * (high - low + 1e-8) + low
        )
        raw_upper = (
            0.5 * (normalized_upper + 1.0) * (high - low + 1e-8) + low
        )
        return {
            "bin_indices": indices.astype(int).tolist(),
            "normalized_zero": normalized_zero.tolist(),
            "raw_lower": raw_lower.tolist(),
            "raw_upper": raw_upper.tolist(),
        }

    # Derive the midpoint between closed and open gripper apertures.
    def _compute_gripper_aperture_threshold(self) -> float:
        # Match the definition used by the DF-05-01 offline collapse diagnostic.
        stats = self.statistics[self.unnorm_key]["proprio"]
        low = np.asarray(stats["q01"], dtype=np.float64)
        high = np.asarray(stats["q99"], dtype=np.float64)
        closed_aperture = low[6] - high[7]
        open_aperture = high[6] - low[7]
        return float(0.5 * (closed_aperture + open_aperture))

    # Apply the official JPEG resize and deterministic evaluation crop.
    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        # Match DF-05-01 preprocessing so model variants see identical pixels.
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB image, got {array.shape}.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        tensor = tf.convert_to_tensor(array, dtype=tf.uint8)
        if bool(self.input_config["jpeg_roundtrip_resize"]):
            tensor = tf.image.encode_jpeg(tensor)
            tensor = tf.io.decode_image(
                tensor,
                expand_animations=False,
                dtype=tf.uint8,
            )
        resolution = int(self.input_config["model_image_resolution"])
        tensor = tf.image.resize(
            tensor,
            (resolution, resolution),
            method="lanczos3",
            antialias=True,
        )
        tensor = tf.cast(
            tf.clip_by_value(tf.round(tensor), 0, 255),
            tf.uint8,
        )

        if bool(self.input_config["center_crop"]):
            crop_scale = float(self.input_config["center_crop_scale"])
            crop_fraction = float(np.sqrt(crop_scale))
            offset = (1.0 - crop_fraction) / 2.0
            box = tf.constant(
                [[offset, offset, offset + crop_fraction, offset + crop_fraction]],
                dtype=tf.float32,
            )
            float_tensor = tf.image.convert_image_dtype(
                tensor,
                tf.float32,
            )[None, ...]
            float_tensor = tf.image.crop_and_resize(
                float_tensor,
                box,
                box_indices=[0],
                crop_size=(resolution, resolution),
            )[0]
            tensor = tf.image.convert_image_dtype(
                tf.clip_by_value(float_tensor, 0.0, 1.0),
                tf.uint8,
                saturate=True,
            )
        return Image.fromarray(tensor.numpy()).convert("RGB")

    # Normalize raw proprio with train-only q01/q99 and clipping.
    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        # Reproduce normalize_action_and_proprio with BOUNDS_Q99 exactly.
        vector = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if vector.shape != (self.proprio_dim,):
            raise ValueError(
                f"Expected proprio shape {(self.proprio_dim,)}, got {vector.shape}."
            )
        if not np.isfinite(vector).all():
            raise ValueError("Proprio contains NaN or Inf.")

        stats = self.statistics[self.unnorm_key]["proprio"]
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(
            stats.get("mask", [True] * self.proprio_dim),
            dtype=bool,
        )
        normalized = np.clip(
            2.0 * (vector - low) / (high - low + 1e-8) - 1.0,
            -1.0,
            1.0,
        )
        normalized = np.where(mask, normalized, vector)
        normalized = np.where(
            np.asarray(stats["min"]) == np.asarray(stats["max"]),
            0.0,
            normalized,
        )
        return normalized.astype(np.float32)

    # Build the exact lowercase OpenVLA prompt used during training.
    def _build_prompt(self, instruction: str) -> str:
        # Reject empty task language before tokenizer invocation.
        normalized = str(instruction).strip().lower()
        if not normalized:
            raise ValueError("Task instruction cannot be empty.")
        return f"In: What action should the robot take to {normalized}?\nOut:"

    # Diagnose zero-equivalent arm actions without modifying the predicted chunk.
    def _diagnose_chunk(
        self,
        action_chunk: np.ndarray,
        raw_proprio: np.ndarray,
    ) -> Dict[str, Any]:
        # Define zero equivalence using the same physical bin intervals as DF-05-01.
        lower = np.asarray(
            self.zero_equivalent_bounds["raw_lower"],
            dtype=np.float32,
        )
        upper = np.asarray(
            self.zero_equivalent_bounds["raw_upper"],
            dtype=np.float32,
        )
        arm_mask = np.logical_and(
            action_chunk[:, :6] >= lower,
            action_chunk[:, :6] <= upper,
        ).all(axis=1)
        current_open = bool(
            float(raw_proprio[6] - raw_proprio[7])
            >= self.gripper_aperture_threshold
        )
        command_open = action_chunk[:, 6] >= 0.5
        full_noop_mask = arm_mask & (command_open == current_open)
        return {
            "zero_equivalent_arm_mask": arm_mask.tolist(),
            "zero_equivalent_full_noop_mask_at_query": full_noop_mask.tolist(),
            "current_gripper_open": current_open,
        }

    # Predict one unnormalized continuous action chunk from the current observation.
    def predict(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        proprio: np.ndarray,
        instruction: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Build the two-image model input and enforce the non-null action-head path.
        if self.action_head is None:
            raise RuntimeError(
                "Continuous action head is missing; discrete fallback is forbidden."
            )
        prompt = self._build_prompt(instruction)
        primary_pil = self._prepare_image(primary_image)
        wrist_pil = self._prepare_image(wrist_image)
        primary_inputs = self.processor(
            prompt,
            primary_pil,
            return_tensors="pt",
        )
        wrist_pixels = self.processor.image_processor(
            wrist_pil,
            return_tensors="pt",
        )["pixel_values"]

        # Match the training collator order: primary channels, then wrist channels.
        pixel_values = torch.cat(
            (primary_inputs["pixel_values"], wrist_pixels),
            dim=1,
        ).to(device=self.device, dtype=self.dtype)
        expected_channels = 6 * int(self.input_config["num_images_in_input"])
        if pixel_values.shape[0] != 1 or pixel_values.shape[1] != expected_channels:
            raise ValueError(
                "Unexpected stacked pixel shape: "
                f"{tuple(pixel_values.shape)}, expected batch=1 channels={expected_channels}."
            )
        input_ids = primary_inputs["input_ids"].to(self.device)
        attention_mask = primary_inputs["attention_mask"].to(self.device)
        normalized_proprio = self._normalize_proprio(proprio)[None, :]

        if bool(self.timing_config["synchronize_cuda_for_measurement"]):
            torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            prediction = self.action_model.predict_action(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                unnorm_key=self.unnorm_key,
                proprio=normalized_proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                noisy_action_projector=None,
                use_film=False,
            )
        if bool(self.timing_config["synchronize_cuda_for_measurement"]):
            torch.cuda.synchronize(self.device)
        inference_latency_ms = (time.perf_counter() - started_at) * 1000.0

        raw_actions = prediction[0] if isinstance(prediction, tuple) else prediction
        action_chunk = np.asarray(raw_actions, dtype=np.float32)
        expected_shape = (self.action_chunk_size, self.action_dim)
        if action_chunk.shape != expected_shape:
            raise ValueError(
                f"Expected continuous action shape {expected_shape}, "
                f"got {action_chunk.shape}."
            )
        if not np.isfinite(action_chunk).all():
            raise ValueError("Predicted continuous action chunk contains NaN or Inf.")

        diagnostics = self._diagnose_chunk(
            action_chunk,
            np.asarray(proprio, dtype=np.float32),
        )
        return action_chunk.copy(), {
            "inference_latency_ms": inference_latency_ms,
            "inference_path": "continuous_l1_action_head",
            "discrete_action_token_decoding_used": False,
            "returned_action_shape": list(action_chunk.shape),
            "returned_chunk_size": int(action_chunk.shape[0]),
            "normalized_proprio": normalized_proprio[0].tolist(),
            **diagnostics,
        }
