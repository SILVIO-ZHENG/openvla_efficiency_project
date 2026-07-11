# Load the DF-04-02 LoRA policy and predict one-step LIBERO actions.

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
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.projectors import ProprioProjector


# Keep TensorFlow image preprocessing on CPU so it cannot reserve model VRAM.
tf.config.set_visible_devices([], "GPU")


# Resolve one supported torch dtype from the YAML value.
def resolve_torch_dtype(value: str) -> torch.dtype:
    normalized = str(value).strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value!r}.")
    return mapping[normalized]


# Load the LoRA policy and predict LIBERO actions.
class LoraPolicy:
    # Initialize the base model, adapter, processor, projector, and statistics.
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.model_config = config["model"]
        self.path_config = config["paths"]
        self.input_config = config["input"]
        self.action_config = config["action"]
        self.timing_config = config["timing"]

        self.device = torch.device(self.model_config["device"])
        self.dtype = resolve_torch_dtype(self.model_config["torch_dtype"])
        self.unnorm_key = "libero_10_no_noops"
        self.action_dim = int(self.action_config["action_dim"])
        self.proprio_dim = int(self.input_config["proprio_dim"])

        self._validate_paths()
        self.statistics = self._load_statistics()
        self.processor = self._load_processor()
        self.model, self.action_model = self._load_model()
        self.proprio_projector = self._load_proprio_projector()

    # Confirm that every model artifact required for inference exists.
    def _validate_paths(self) -> None:
        required_files = {
            "base model config": Path(self.path_config["base_model_path"]) / "config.json",
            "adapter config": Path(self.path_config["adapter_path"]) / "adapter_config.json",
            "adapter weights": Path(self.path_config["adapter_path"]) / "adapter_model.safetensors",
            "proprio projector": Path(self.path_config["proprio_projector_path"]),
            "dataset statistics": Path(self.path_config["dataset_statistics_path"]),
            "processor config": Path(self.path_config["processor_path"]) / "preprocessor_config.json",
            "tokenizer config": Path(self.path_config["processor_path"]) / "tokenizer_config.json",
        }
        missing = [name for name, path in required_files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing inference artifacts: " + ", ".join(missing))

    # Load the train-only normalization statistics used by DF-04-02.
    def _load_statistics(self) -> Dict[str, Any]:
        path = Path(self.path_config["dataset_statistics_path"])
        with path.open("r", encoding="utf-8") as handle:
            statistics = json.load(handle)
        if self.unnorm_key not in statistics:
            raise KeyError(f"Missing statistics key: {self.unnorm_key!r}.")
        return statistics

    # Reconstruct the saved processor without requiring processor_config.json.
    def _load_processor(self) -> PrismaticProcessor:
        processor_path = Path(self.path_config["processor_path"])
        image_processor = PrismaticImageProcessor.from_pretrained(
            processor_path,
            local_files_only=bool(self.model_config["local_files_only"]),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            processor_path,
            local_files_only=bool(self.model_config["local_files_only"]),
        )
        return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Load OpenVLA and attach the trained LoRA adapter.
    def _load_model(self) -> Tuple[PeftModel, OpenVLAForActionPrediction]:
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

        # The training source retains an eight-step inference constant from OFT.
        modeling_prismatic.NUM_ACTIONS_CHUNK = int(
            self.action_config["action_chunk_size"]
        )
        if modeling_prismatic.NUM_ACTIONS_CHUNK != 1:
            raise ValueError("DF-04-02 inference requires NUM_ACTIONS_CHUNK=1.")

        return model, action_model

    # Recreate the 8-D proprio projector and load its trained state.
    def _load_proprio_projector(self) -> ProprioProjector:
        projector = ProprioProjector(
            llm_dim=int(self.action_model.llm_dim),
            proprio_dim=self.proprio_dim,
        )
        state_dict = torch.load(
            self.path_config["proprio_projector_path"],
            map_location="cpu",
            weights_only=True,
        )
        projector.load_state_dict(state_dict, strict=True)
        projector.to(device=self.device, dtype=self.dtype)
        projector.eval()
        return projector

    # Apply the official LIBERO resize and deterministic evaluation crop.
    def _prepare_image(self, image: np.ndarray) -> Image.Image:
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
            float_tensor = tf.image.convert_image_dtype(tensor, tf.float32)[None, ...]
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

    # Normalize the raw 8-D proprio vector with train-only q01/q99 statistics.
    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        vector = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if vector.shape != (self.proprio_dim,):
            raise ValueError(
                f"Expected proprio shape {(self.proprio_dim,)}, got {vector.shape}."
            )
        if not np.isfinite(vector).all():
            raise ValueError("Proprio contains NaN or Inf.")

        statistics = self.statistics[self.unnorm_key]["proprio"]
        low = np.asarray(statistics["q01"], dtype=np.float32)
        high = np.asarray(statistics["q99"], dtype=np.float32)
        mask = np.asarray(statistics.get("mask", [True] * self.proprio_dim), dtype=bool)
        normalized = np.clip(
            2.0 * (vector - low) / (high - low + 1e-8) - 1.0,
            -1.0,
            1.0,
        )
        normalized = np.where(mask, normalized, vector)
        normalized = np.where(
            np.asarray(statistics["min"]) == np.asarray(statistics["max"]),
            0.0,
            normalized,
        )
        return normalized.astype(np.float32)

    # Build the exact OpenVLA prompt used by the training pipeline.
    def _build_prompt(self, instruction: str) -> str:
        normalized = str(instruction).strip().lower()
        if not normalized:
            raise ValueError("Task instruction cannot be empty.")
        return f"In: What action should the robot take to {normalized}?\nOut:"

    # Predict one continuous 7-D action from the current observation.
    def predict(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        proprio: np.ndarray,
        instruction: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        prompt = self._build_prompt(instruction)
        primary_pil = self._prepare_image(primary_image)
        wrist_pil = self._prepare_image(wrist_image)

        primary_inputs = self.processor(prompt, primary_pil, return_tensors="pt")
        wrist_pixels = self.processor.image_processor(
            wrist_pil,
            return_tensors="pt",
        )["pixel_values"]

        # Match the training collator order: primary channels, then wrist channels.
        pixel_values = torch.cat(
            (primary_inputs["pixel_values"], wrist_pixels),
            dim=1,
        ).to(device=self.device, dtype=self.dtype)
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
                action_head=None,
                noisy_action_projector=None,
                use_film=False,
                do_sample=False,
            )

        if bool(self.timing_config["synchronize_cuda_for_measurement"]):
            torch.cuda.synchronize(self.device)
        inference_latency_ms = (time.perf_counter() - started_at) * 1000.0

        raw_actions = prediction[0] if isinstance(prediction, tuple) else prediction
        action_chunk = np.asarray(raw_actions, dtype=np.float32)
        expected_shape = (1, self.action_dim)
        if action_chunk.shape != expected_shape:
            raise ValueError(
                f"Expected one-step action shape {expected_shape}, got {action_chunk.shape}."
            )

        action = action_chunk[0].copy()
        if not np.isfinite(action).all():
            raise ValueError("Predicted action contains NaN or Inf.")

        return action, {
            "inference_latency_ms": inference_latency_ms,
            "returned_action_shape": list(action_chunk.shape),
            "returned_chunk_size": int(action_chunk.shape[0]),
            "normalized_proprio": normalized_proprio[0].tolist(),
        }
