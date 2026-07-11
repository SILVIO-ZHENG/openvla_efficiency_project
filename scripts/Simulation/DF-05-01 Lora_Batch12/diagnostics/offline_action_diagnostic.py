# Diagnose DF-04-02 action-bin predictions on fixed validation transitions.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ACTION_DIM = 7
PROPRIO_DIM = 8
IGNORE_INDEX = -100
DIMENSION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


# Parse diagnostic options with safe defaults for a 32 GB GPU.
def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the DF-04-02 offline validation action diagnostic."
    )
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=project_root / "config" / "simulation_config.yaml",
    )
    parser.add_argument("--runtime-config", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--split-name",
        choices=("train", "val"),
        default="val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "flash_attention_2", "sdpa"),
        default="eager",
    )
    return parser.parse_args()


# Load one YAML or JSON file and require a mapping at the root.
def load_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"File root must be a mapping: {path}")
    return value


# Resolve one absolute or project-relative path.
def resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base_dir / path).resolve()


# Compute a file digest for output provenance.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Seed all random generators used before TensorFlow dataset construction.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Extract exactly seven action predictions and the separate stop prediction.
def extract_action_tokens(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,S,V] and labels [B,S].")
    if logits.shape[0] != labels.shape[0] or logits.shape[1] < labels.shape[1]:
        raise ValueError("Logits and labels cannot be aligned.")

    text_offset = logits.shape[1] - labels.shape[1]
    shifted_logits = logits[:, text_offset:, :][:, :-1, :]
    shifted_labels = labels[:, 1:]
    predicted_ids = shifted_logits.argmax(dim=-1)
    true_actions = []
    pred_actions = []
    true_stops = []
    pred_stops = []

    for batch_index in range(labels.shape[0]):
        positions = shifted_labels[batch_index].ne(IGNORE_INDEX).nonzero().flatten()
        if positions.numel() != ACTION_DIM + 1:
            raise RuntimeError(
                f"Sample {batch_index} has {positions.numel()} supervised positions; "
                "expected 7 action tokens plus 1 stop token."
            )
        action_positions = positions[:ACTION_DIM]
        stop_position = positions[-1]
        true_actions.append(shifted_labels[batch_index, action_positions])
        pred_actions.append(predicted_ids[batch_index, action_positions])
        true_stops.append(shifted_labels[batch_index, stop_position])
        pred_stops.append(predicted_ids[batch_index, stop_position])

    return (
        torch.stack(true_actions).cpu().numpy(),
        torch.stack(pred_actions).cpu().numpy(),
        torch.stack(true_stops).cpu().numpy(),
        torch.stack(pred_stops).cpu().numpy(),
    )


# Decode OpenVLA token ids into zero-based action-bin indices.
def token_ids_to_bins(
    token_ids: np.ndarray,
    action_vocab_size: int,
    number_of_centers: int,
) -> np.ndarray:
    indices = action_vocab_size - np.asarray(token_ids, dtype=np.int64) - 1
    return np.clip(indices, 0, number_of_centers - 1).astype(np.int64)


# Quantize normalized values with the same OpenVLA boundaries.
def normalized_to_bins(
    values: np.ndarray,
    boundaries: np.ndarray,
    number_of_centers: int,
) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
    indices = np.digitize(clipped, boundaries) - 1
    return np.clip(indices, 0, number_of_centers - 1).astype(np.int64)


# Undo q01/q99 normalization while respecting the statistics mask.
def unnormalize(values: np.ndarray, statistics: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = np.asarray(statistics["q01"], dtype=np.float64)
    high = np.asarray(statistics["q99"], dtype=np.float64)
    mask = np.asarray(statistics.get("mask", np.ones_like(low)), dtype=bool)
    raw = 0.5 * (values + 1.0) * (high - low + 1e-8) + low
    return np.where(mask, raw, values)


# Find each dimension's bin containing raw physical action zero.
def compute_zero_bins(
    action_statistics: Mapping[str, Any],
    boundaries: np.ndarray,
    number_of_centers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    low = np.asarray(action_statistics["q01"], dtype=np.float64)
    high = np.asarray(action_statistics["q99"], dtype=np.float64)
    mask = np.asarray(action_statistics["mask"], dtype=bool)
    normalized_zero = np.where(
        mask,
        np.clip(2.0 * (0.0 - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0),
        0.0,
    )
    return (
        normalized_to_bins(normalized_zero, boundaries, number_of_centers),
        normalized_zero,
    )


# Write rows to one CSV file with a stable header.
def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Run the complete diagnostic without starting LIBERO or MuJoCo.
def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    project_root = Path(__file__).resolve().parents[1]
    simulation_config_path = args.simulation_config.expanduser().resolve()
    simulation_config = load_mapping(simulation_config_path)
    simulation_config["model"]["attn_implementation"] = args.attn_implementation
    openvla_source_root = resolve_path(
        simulation_config["paths"]["openvla_source_root"], project_root
    )
    for import_path in (project_root, openvla_source_root):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    runtime_config_path = (
        args.runtime_config.expanduser().resolve()
        if args.runtime_config is not None
        else resolve_path(simulation_config["paths"]["training_config_path"], project_root)
    )
    runtime_config = load_mapping(runtime_config_path)
    dataset_config = runtime_config["dataset"]
    split_json_path = (
        args.split_json.expanduser().resolve()
        if args.split_json is not None
        else resolve_path(dataset_config["episode_split_json"], project_root)
    )
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else resolve_path(dataset_config["data_root_dir"], project_root)
    )
    dataset_statistics_path = resolve_path(
        simulation_config["paths"]["dataset_statistics_path"], project_root
    )
    adapter_path = resolve_path(simulation_config["paths"]["adapter_path"], project_root)
    proprio_projector_path = resolve_path(
        simulation_config["paths"]["proprio_projector_path"], project_root
    )
    required_paths = {
        "simulation_config": simulation_config_path,
        "runtime_config": runtime_config_path,
        "split_json": split_json_path,
        "dataset_statistics": dataset_statistics_path,
        "adapter_config": adapter_path / "adapter_config.json",
        "adapter_weights": adapter_path / "adapter_model.safetensors",
        "proprio_projector": proprio_projector_path,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.is_file()]
    if not data_root.is_dir():
        missing.append(f"data_root: {data_root}")
    if missing:
        raise FileNotFoundError("Missing diagnostic inputs:\n" + "\n".join(missing))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENVLA_EPISODE_SPLIT_JSON"] = str(split_json_path)
    os.environ["OPENVLA_EPISODE_SPLIT_NAME"] = args.split_name
    seed_everything(args.seed)

    from src.lora_policy import LoraPolicy
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

    import tensorflow as tf

    tf.random.set_seed(args.seed)
    started_at = time.perf_counter()
    print(
        "Loading DF-04-02 best adapter and fixed "
        f"{args.split_name} split..."
    )
    policy = LoraPolicy(simulation_config)
    action_tokenizer = ActionTokenizer(policy.processor.tokenizer)
    transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=policy.processor.tokenizer,
        image_transform=policy.processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        predict_stop_token=True,
        use_wrist_image=True,
        use_proprio=True,
    )
    diagnostic_dataset = RLDSDataset(
        data_root_dir=data_root,
        data_mix=str(dataset_config["name"]),
        batch_transform=transform,
        resize_resolution=tuple(policy.action_model.config.image_sizes),
        shuffle_buffer_size=1,
        train=args.split_name == "train",
        image_aug=False,
        use_wrist_image=True,
        use_proprio=True,
        normalization_stats=policy.statistics,
    )
    collator = PaddedCollatorForActionPrediction(
        model_max_length=policy.processor.tokenizer.model_max_length,
        pad_token_id=policy.processor.tokenizer.pad_token_id,
        padding_side="right",
        use_wrist_image=True,
        use_proprio=True,
        validate_finite_values=True,
        expected_action_chunk_len=1,
        expected_action_dim=ACTION_DIM,
        expected_proprio_dim=PROPRIO_DIM,
    )
    loader = DataLoader(
        diagnostic_dataset,
        batch_size=args.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    if len(diagnostic_dataset) < args.num_samples:
        raise ValueError(
            f"{args.split_name} pipeline has {len(diagnostic_dataset)} transitions, "
            f"fewer than requested {args.num_samples}."
        )

    action_statistics = policy.statistics[policy.unnorm_key]["action"]
    proprio_statistics = policy.statistics[policy.unnorm_key]["proprio"]
    boundaries = np.asarray(policy.action_model.bins, dtype=np.float64)
    centers = np.asarray(policy.action_model.bin_centers, dtype=np.float64)
    number_of_centers = int(centers.shape[0])
    action_vocab_size = int(policy.action_model.vocab_size)
    zero_bins, normalized_zero = compute_zero_bins(
        action_statistics, boundaries, number_of_centers
    )

    proprio_low = np.asarray(proprio_statistics["q01"], dtype=np.float64)
    proprio_high = np.asarray(proprio_statistics["q99"], dtype=np.float64)
    closed_aperture = proprio_low[6] - proprio_high[7]
    open_aperture = proprio_high[6] - proprio_low[7]
    gripper_aperture_threshold = float(0.5 * (closed_aperture + open_aperture))
    valid_action_token_min = action_vocab_size - number_of_centers
    valid_action_token_max = action_vocab_size - 1

    transition_rows: List[Dict[str, Any]] = []
    collected: Dict[str, List[np.ndarray]] = {
        name: []
        for name in (
            "true_bins",
            "pred_bins",
            "true_ids",
            "pred_ids",
            "true_stops",
            "pred_stops",
            "token_correct",
            "bin_correct",
            "pred_valid",
            "true_zero",
            "pred_zero",
            "true_arm_noop",
            "pred_arm_noop",
            "true_full_noop",
            "pred_full_noop",
            "raw_error",
            "normalized_error",
            "label_bin_consistent",
        )
    }
    evaluated = 0
    policy.model.eval()
    policy.proprio_projector.eval()

    for batch in loader:
        remaining = args.num_samples - evaluated
        if remaining <= 0:
            break
        current_batch_size = min(int(batch["labels"].shape[0]), remaining)
        if current_batch_size < int(batch["labels"].shape[0]):
            batch = {
                key: value[:current_batch_size]
                if isinstance(value, (torch.Tensor, list))
                else value
                for key, value in batch.items()
            }

        labels = batch["labels"].to(policy.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=policy.dtype
        ):
            output = policy.model(
                input_ids=batch["input_ids"].to(policy.device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(
                    policy.device, non_blocking=True
                ),
                pixel_values=batch["pixel_values"].to(
                    device=policy.device,
                    dtype=policy.dtype,
                    non_blocking=True,
                ),
                labels=labels,
                proprio=batch["proprio"].to(
                    device=policy.device,
                    dtype=policy.dtype,
                    non_blocking=True,
                ),
                proprio_projector=policy.proprio_projector,
                use_film=False,
            )
        if output.logits is None:
            raise RuntimeError("Model forward did not return logits.")

        true_ids, pred_ids, true_stops, pred_stops = extract_action_tokens(
            output.logits, labels
        )
        true_bins = token_ids_to_bins(true_ids, action_vocab_size, number_of_centers)
        pred_bins = token_ids_to_bins(pred_ids, action_vocab_size, number_of_centers)
        true_normalized = batch["actions"][:, 0, :].cpu().numpy().astype(np.float64)
        pred_normalized = centers[pred_bins]
        true_raw = unnormalize(true_normalized, action_statistics)
        pred_raw = unnormalize(pred_normalized, action_statistics)
        token_correct = true_ids == pred_ids
        bin_correct = true_bins == pred_bins
        pred_valid = (pred_ids >= valid_action_token_min) & (
            pred_ids <= valid_action_token_max
        )
        true_zero = true_bins == zero_bins[None, :]
        pred_zero = pred_bins == zero_bins[None, :]
        true_arm_noop = true_zero[:, :6].all(axis=1)
        pred_arm_noop = pred_zero[:, :6].all(axis=1)

        raw_proprio = unnormalize(
            batch["proprio"].cpu().numpy().astype(np.float64),
            proprio_statistics,
        )
        current_gripper_open = (
            raw_proprio[:, 6] - raw_proprio[:, 7]
        ) >= gripper_aperture_threshold
        true_gripper_no_change = (true_raw[:, 6] >= 0.5) == current_gripper_open
        pred_gripper_no_change = (pred_raw[:, 6] >= 0.5) == current_gripper_open
        true_full_noop = true_arm_noop & true_gripper_no_change
        pred_full_noop = pred_arm_noop & pred_gripper_no_change
        raw_error = np.abs(pred_raw - true_raw)
        normalized_error = np.abs(pred_normalized - true_normalized)
        quantized_true_bins = normalized_to_bins(
            true_normalized, boundaries, number_of_centers
        )
        label_bin_consistent = true_bins == quantized_true_bins

        batch_arrays = {
            "true_bins": true_bins,
            "pred_bins": pred_bins,
            "true_ids": true_ids,
            "pred_ids": pred_ids,
            "true_stops": true_stops,
            "pred_stops": pred_stops,
            "token_correct": token_correct,
            "bin_correct": bin_correct,
            "pred_valid": pred_valid,
            "true_zero": true_zero,
            "pred_zero": pred_zero,
            "true_arm_noop": true_arm_noop,
            "pred_arm_noop": pred_arm_noop,
            "true_full_noop": true_full_noop,
            "pred_full_noop": pred_full_noop,
            "raw_error": raw_error,
            "normalized_error": normalized_error,
            "label_bin_consistent": label_bin_consistent,
        }
        for name, values in batch_arrays.items():
            collected[name].append(values)

        dataset_names = batch.get("dataset_names", [""] * current_batch_size)
        for row_index in range(current_batch_size):
            row: Dict[str, Any] = {
                "sample_index": evaluated + row_index,
                "dataset_name": dataset_names[row_index],
            }
            for dimension, dimension_name in enumerate(DIMENSION_NAMES):
                row[f"true_bin_{dimension_name}"] = int(true_bins[row_index, dimension])
                row[f"pred_bin_{dimension_name}"] = int(pred_bins[row_index, dimension])
                row[f"true_token_id_{dimension_name}"] = int(
                    true_ids[row_index, dimension]
                )
                row[f"pred_token_id_{dimension_name}"] = int(
                    pred_ids[row_index, dimension]
                )
                row[f"token_correct_{dimension_name}"] = bool(
                    token_correct[row_index, dimension]
                )
                row[f"pred_zero_bin_{dimension_name}"] = bool(
                    pred_zero[row_index, dimension]
                )
                row[f"true_action_raw_{dimension_name}"] = float(
                    true_raw[row_index, dimension]
                )
                row[f"pred_action_raw_{dimension_name}"] = float(
                    pred_raw[row_index, dimension]
                )
            row.update(
                {
                    "true_stop_token_id": int(true_stops[row_index]),
                    "pred_stop_token_id": int(pred_stops[row_index]),
                    "stop_correct": bool(true_stops[row_index] == pred_stops[row_index]),
                    "action_token_accuracy_excluding_stop": float(
                        token_correct[row_index].mean()
                    ),
                    "whole_action_accuracy": bool(bin_correct[row_index].all()),
                    "true_arm_noop": bool(true_arm_noop[row_index]),
                    "pred_arm_noop": bool(pred_arm_noop[row_index]),
                    "true_full_noop": bool(true_full_noop[row_index]),
                    "pred_full_noop": bool(pred_full_noop[row_index]),
                    "current_gripper_open": bool(current_gripper_open[row_index]),
                    "action_l1_raw_7d": float(raw_error[row_index].mean()),
                    "action_l1_raw_arm6": float(raw_error[row_index, :6].mean()),
                    "all_pred_action_tokens_valid": bool(pred_valid[row_index].all()),
                    "label_bin_consistent": bool(label_bin_consistent[row_index].all()),
                }
            )
            transition_rows.append(row)

        evaluated += current_batch_size
        if evaluated % 25 == 0 or evaluated == args.num_samples:
            print(
                f"Processed {evaluated}/{args.num_samples} "
                f"{args.split_name} transitions."
            )
        del output, labels

    if evaluated != args.num_samples:
        raise RuntimeError(
            f"{args.split_name} iterator ended at "
            f"{evaluated}/{args.num_samples} samples."
        )

    metrics = {
        name: np.concatenate(parts, axis=0) for name, parts in collected.items()
    }
    dimension_rows = []
    for dimension, dimension_name in enumerate(DIMENSION_NAMES):
        dimension_rows.append(
            {
                "dimension_index": dimension,
                "dimension_name": dimension_name,
                "is_motion_dimension": dimension < 6,
                "physical_zero_bin_index": int(zero_bins[dimension]),
                "physical_zero_normalized": float(normalized_zero[dimension]),
                "true_zero_bin_rate": float(metrics["true_zero"][:, dimension].mean()),
                "pred_zero_bin_rate": float(metrics["pred_zero"][:, dimension].mean()),
                "token_accuracy_excluding_stop": float(
                    metrics["token_correct"][:, dimension].mean()
                ),
                "bin_accuracy_excluding_stop": float(
                    metrics["bin_correct"][:, dimension].mean()
                ),
                "pred_valid_action_token_rate": float(
                    metrics["pred_valid"][:, dimension].mean()
                ),
                "action_l1_raw": float(metrics["raw_error"][:, dimension].mean()),
                "action_l1_normalized": float(
                    metrics["normalized_error"][:, dimension].mean()
                ),
            }
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            project_root
            / "output"
            / "diagnostics"
            / f"offline_action_{args.split_name}_{args.num_samples}"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = output_dir / "transition_predictions.csv"
    dimension_path = output_dir / "dimension_summary.csv"
    summary_path = output_dir / "summary.json"
    write_csv(transition_path, transition_rows)
    write_csv(dimension_path, dimension_rows)

    token_correct = metrics["token_correct"]
    bin_correct = metrics["bin_correct"]
    summary = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic": "DF-05-01 offline action diagnostic",
        "run_id": runtime_config.get("run_id_note"),
        "model_variant": "DF-04-02 LoRA Batch12",
        "checkpoint_step": int(policy.model_config["checkpoint_step"]),
        "split_name": args.split_name,
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "evaluated_samples": evaluated,
        "split_transition_count_reported_by_pipeline": len(diagnostic_dataset),
        "batch_size": args.batch_size,
        "attention_implementation": args.attn_implementation,
        "torch_dtype": str(policy.dtype),
        "device": str(policy.device),
        "elapsed_seconds": time.perf_counter() - started_at,
        "paths": {
            **{name: str(path) for name, path in required_paths.items()},
            "data_root": str(data_root),
        },
        "sha256": {
            name: sha256_file(path) for name, path in required_paths.items()
        },
        "definitions": {
            "zero_bin": (
                "Per-dimension OpenVLA bin containing raw physical action zero; "
                "not an assumed geometric center bin."
            ),
            "arm_noop": "All first six motion dimensions equal their zero bins.",
            "full_noop": (
                "arm_noop plus a gripper command matching the current open/closed "
                "state estimated from raw gripper qpos."
            ),
            "action_token_accuracy_excluding_stop": (
                "Exact token accuracy over the seven action positions only."
            ),
            "whole_action_accuracy": (
                "Fraction of samples where all seven predicted action bins match."
            ),
            "action_l1_raw": "Mean absolute error after q01/q99 unnormalization.",
            "gripper_aperture_threshold": gripper_aperture_threshold,
        },
        "zero_action": {
            "dimension_names": list(DIMENSION_NAMES),
            "zero_bin_indices": zero_bins.tolist(),
            "zero_normalized_values": normalized_zero.tolist(),
            "true_zero_bin_rate_per_dimension": metrics["true_zero"].mean(axis=0).tolist(),
            "pred_zero_bin_rate_per_dimension": metrics["pred_zero"].mean(axis=0).tolist(),
            "true_arm_noop_rate": float(metrics["true_arm_noop"].mean()),
            "pred_arm_noop_rate": float(metrics["pred_arm_noop"].mean()),
            "true_full_noop_rate": float(metrics["true_full_noop"].mean()),
            "pred_full_noop_rate": float(metrics["pred_full_noop"].mean()),
        },
        "accuracy": {
            "per_dimension_token_accuracy_excluding_stop": token_correct.mean(axis=0).tolist(),
            "per_dimension_bin_accuracy_excluding_stop": bin_correct.mean(axis=0).tolist(),
            "action_token_accuracy_excluding_stop": float(token_correct.mean()),
            "action_bin_accuracy_excluding_stop": float(bin_correct.mean()),
            "stop_token_accuracy": float(
                (metrics["true_stops"] == metrics["pred_stops"]).mean()
            ),
            "whole_action_token_accuracy": float(token_correct.all(axis=1).mean()),
            "whole_action_accuracy": float(bin_correct.all(axis=1).mean()),
            "pred_valid_action_token_rate": float(metrics["pred_valid"].mean()),
            "label_bin_consistency_rate": float(
                metrics["label_bin_consistent"].mean()
            ),
        },
        "error": {
            "action_l1_raw_7d": float(metrics["raw_error"].mean()),
            "action_l1_raw_arm6": float(metrics["raw_error"][:, :6].mean()),
            "action_l1_raw_gripper": float(metrics["raw_error"][:, 6].mean()),
            "action_l1_normalized_7d": float(metrics["normalized_error"].mean()),
            "action_l1_raw_per_dimension": metrics["raw_error"].mean(axis=0).tolist(),
        },
        "outputs": {
            "transition_predictions_csv": str(transition_path),
            "dimension_summary_csv": str(dimension_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nDF-05-01 offline action diagnostic: COMPLETED")
    print(f"evaluated_samples={evaluated}")
    print(
        "action_token_accuracy_excluding_stop="
        f"{summary['accuracy']['action_token_accuracy_excluding_stop']:.6f}"
    )
    print(
        f"whole_action_accuracy={summary['accuracy']['whole_action_accuracy']:.6f}"
    )
    print(f"pred_arm_noop_rate={summary['zero_action']['pred_arm_noop_rate']:.6f}")
    print(f"pred_full_noop_rate={summary['zero_action']['pred_full_noop_rate']:.6f}")
    print(f"action_l1_raw_7d={summary['error']['action_l1_raw_7d']:.6f}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()