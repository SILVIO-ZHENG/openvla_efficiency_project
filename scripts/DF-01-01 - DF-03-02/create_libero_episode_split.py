# Notes:
# The following default paths are local paths on the author's cloud machine.
# They are recorded for experiment reproducibility and do not contain access credentials.
#
# DF-03-02 LIBERO Episode Split Script
#
# This script creates a fixed 80/10/10 train/val/test episode split for the
# downloaded OpenVLA modified LIBERO RLDS dataset.
#
# Important design:
# - The original RLDS / TFDS dataset is NOT copied, moved, or modified.
# - Each LIBERO sub-dataset is split separately.
# - One JSON file is used as the single source of truth for all later loaders.
# - The Markdown log is only a human-readable record.
#
# Expected cloud dataset root:
# /root/autodl-tmp/datasets/openvla_modified_libero_rlds
#
# Expected outputs:
# results/splits/df_03_02_libero_episode_split_seed42.json
# logs/df_03_02_libero_episode_split_log.md

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# The four sub-dataset names are the confirmed folders in openvla/modified_libero_rlds.
DEFAULT_SUB_DATASETS: List[str] = [
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
]


# This mapping records the task type of each LIBERO sub-dataset.
TASK_SUITE_MAP: Dict[str, str] = {
    "libero_spatial_no_noops": "spatial",
    "libero_object_no_noops": "object",
    "libero_goal_no_noops": "goal",
    "libero_10_no_noops": "libero_10",
}


# This function reads command-line settings for dataset root, split seed, and output paths.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed per-package 80/10/10 episode split for modified LIBERO RLDS."
    )

    # This path points to the downloaded dataset on the author's cloud machine.
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/root/autodl-tmp/datasets/openvla_modified_libero_rlds",
        help=(
            "Root path of the downloaded modified LIBERO RLDS dataset. "
            "This is a cloud-machine local path, not a credential."
        ),
    )

    # The source split is train because the downloaded dataset currently confirms train episodes.
    parser.add_argument(
        "--source-split",
        type=str,
        default="train",
        help="Source split used to create the local train/val/test episode split.",
    )

    # The seed fixes the shuffled episode order so all later experiments use the same samples.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic episode splitting.",
    )

    # The train ratio controls the first part of the split inside each sub-dataset.
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for each LIBERO sub-dataset.",
    )

    # The validation ratio controls the second part of the split inside each sub-dataset.
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for each LIBERO sub-dataset.",
    )

    # The test ratio is checked for documentation; the real test count receives the remainder.
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio label for documentation. The final test count receives the remainder.",
    )

    # This JSON file is the single source of truth for later fixed sample selection.
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/splits/df_03_02_libero_episode_split_seed42.json",
        help="Path for the generated episode split JSON file.",
    )

    # This Markdown file records the split method and per-dataset counts.
    parser.add_argument(
        "--output-log",
        type=str,
        default="logs/df_03_02_libero_episode_split_log.md",
        help="Path for the generated episode split Markdown log.",
    )

    return parser.parse_args()


# This function checks whether the requested split ratios are valid.
def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Train, val, and test ratios must all be positive.")

    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Train, val, and test ratios must sum to 1.0, but got {ratio_sum}."
        )


# This function creates a stable integer seed for each sub-dataset.
def make_stable_sub_dataset_seed(seed: int, sub_dataset_name: str) -> int:
    seed_text = f"{seed}:{sub_dataset_name}"

    # SHA256 avoids Python hash randomization and makes the seed stable across runs.
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()

    # The first 16 hex characters are enough for a stable deterministic seed.
    return int(digest[:16], 16)


# This function loads a JSON file from disk.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# This function finds the TFDS version folder, such as 1.0.0, under each sub-dataset.
def find_dataset_version_dir(sub_dataset_dir: Path) -> Optional[Path]:
    if not sub_dataset_dir.exists():
        return None

    # A valid TFDS version folder should contain dataset_info.json.
    for child in sorted(sub_dataset_dir.iterdir()):
        if child.is_dir() and (child / "dataset_info.json").exists():
            return child

    return None


# This function converts TFDS shard length values into integers safely.
def safe_int_sum(values: Any) -> int:
    if not isinstance(values, list):
        return 0

    total = 0

    # Some dataset_info.json files store shard lengths as strings.
    for value in values:
        try:
            total += int(value)
        except Exception:
            continue

    return total


# This function extracts the episode count for the selected source split.
def extract_split_count(dataset_info: Dict[str, Any], split_name: str) -> int:
    splits = dataset_info.get("splits", [])

    # This branch handles split metadata stored as a list.
    if isinstance(splits, list):
        for split in splits:
            if not isinstance(split, dict):
                continue

            if split.get("name") != split_name:
                continue

            if "numExamples" in split:
                return int(split["numExamples"])

            if "num_examples" in split:
                return int(split["num_examples"])

            if "shardLengths" in split:
                return safe_int_sum(split["shardLengths"])

            if "shard_lengths" in split:
                return safe_int_sum(split["shard_lengths"])

    # This branch handles split metadata stored as a dictionary.
    if isinstance(splits, dict):
        split = splits.get(split_name, {})

        if isinstance(split, dict):
            if "numExamples" in split:
                return int(split["numExamples"])

            if "num_examples" in split:
                return int(split["num_examples"])

            if "shardLengths" in split:
                return safe_int_sum(split["shardLengths"])

            if "shard_lengths" in split:
                return safe_int_sum(split["shard_lengths"])

    return 0


# This function creates episode indices from 0 to episode_count - 1.
def build_episode_indices(episode_count: int) -> List[int]:
    return list(range(episode_count))


# This function creates a deterministic 80/10/10 split for one sub-dataset.
def split_episode_indices(
    episode_indices: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    sub_dataset_name: str,
) -> Dict[str, List[int]]:
    # Each package uses its own stable seed, so the packages are deterministic but not identical.
    local_seed = make_stable_sub_dataset_seed(seed=seed, sub_dataset_name=sub_dataset_name)

    # A copied list is shuffled so the original list is not modified outside this function.
    shuffled_indices = list(episode_indices)
    random.Random(local_seed).shuffle(shuffled_indices)

    total_count = len(shuffled_indices)

    # Train and val use integer floor counts.
    # The test split receives the remainder so no episode is lost.
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    train_indices = shuffled_indices[:train_count]
    val_indices = shuffled_indices[train_count : train_count + val_count]
    test_indices = shuffled_indices[train_count + val_count :]

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


# This function inspects one sub-dataset and builds its split record.
def create_sub_dataset_split(
    dataset_root: Path,
    sub_dataset_name: str,
    source_split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Any]:
    sub_dataset_dir = dataset_root / sub_dataset_name
    version_dir = find_dataset_version_dir(sub_dataset_dir)

    if version_dir is None:
        raise FileNotFoundError(
            f"Version directory with dataset_info.json was not found for {sub_dataset_name}."
        )

    dataset_info_path = version_dir / "dataset_info.json"
    dataset_info = load_json(dataset_info_path)
    episode_count = extract_split_count(dataset_info, source_split)

    if episode_count <= 0:
        raise ValueError(
            f"No valid episode count was found for {sub_dataset_name} split={source_split}."
        )

    episode_indices = build_episode_indices(episode_count)

    split_indices = split_episode_indices(
        episode_indices=episode_indices,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        sub_dataset_name=sub_dataset_name,
    )

    task_suite = TASK_SUITE_MAP.get(sub_dataset_name, "unknown")

    # The split record stores only lightweight episode indices, not dataset content.
    return {
        "sub_dataset": sub_dataset_name,
        "task_suite": task_suite,
        "sub_dataset_dir": str(sub_dataset_dir),
        "version_dir": str(version_dir),
        "dataset_info_path": str(dataset_info_path),
        "source_split": source_split,
        "original_episode_count": episode_count,
        "train_count": len(split_indices["train"]),
        "val_count": len(split_indices["val"]),
        "test_count": len(split_indices["test"]),
        "train_episode_indices": split_indices["train"],
        "val_episode_indices": split_indices["val"],
        "test_episode_indices": split_indices["test"],
    }


# This function creates all sub-dataset split records and total counts.
def create_all_splits(
    dataset_root: Path,
    source_split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, Any]:
    validate_split_ratios(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    sub_dataset_records: List[Dict[str, Any]] = []

    # Each sub-dataset is split separately.
    # This is a per-package split, not a global mixed split over all 1693 episodes.
    for sub_dataset_name in DEFAULT_SUB_DATASETS:
        record = create_sub_dataset_split(
            dataset_root=dataset_root,
            sub_dataset_name=sub_dataset_name,
            source_split=source_split,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        sub_dataset_records.append(record)

    total_original = sum(record["original_episode_count"] for record in sub_dataset_records)
    total_train = sum(record["train_count"] for record in sub_dataset_records)
    total_val = sum(record["val_count"] for record in sub_dataset_records)
    total_test = sum(record["test_count"] for record in sub_dataset_records)

    # The full split object records both the method and the actual episode indices.
    return {
        "task_id": "DF-03-02",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "openvla/modified_libero_rlds",
        "dataset_root": str(dataset_root),
        "source_split": source_split,
        "random_seed": seed,
        "split_scope": "per_sub_dataset",
        "single_source_of_truth": True,
        "split_file_role": (
            "This JSON is the only official split source. "
            "Training, validation, testing, and later experiment loaders should read this JSON."
        ),
        "physical_data_copy": False,
        "split_rule": (
            "Each LIBERO sub-dataset is split independently. "
            "For each sub-dataset, episode indices are shuffled with a deterministic package-specific seed. "
            "train_count=int(total*0.8), val_count=int(total*0.1), "
            "and test receives the remaining episodes."
        ),
        "requested_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "totals": {
            "original_episode_count": total_original,
            "train_count": total_train,
            "val_count": total_val,
            "test_count": total_test,
        },
        "sub_datasets": sub_dataset_records,
    }


# This function writes the full split JSON for later fixed sample selection.
def write_json(output_json: Path, split_data: Dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=2, ensure_ascii=False)


# This function writes a Markdown log that records original counts, split counts, and saved paths.
def write_markdown_log(output_log: Path, split_data: Dict[str, Any], output_json: Path) -> None:
    output_log.parent.mkdir(parents=True, exist_ok=True)

    totals = split_data["totals"]
    ratios = split_data["requested_ratios"]

    lines: List[str] = []

    lines.append("# DF-03-02 LIBERO Episode Split Log\n\n")

    lines.append("## Notes\n\n")
    lines.append("The following paths are local paths on the author's cloud machine.\n")
    lines.append("They are recorded for experiment reproducibility and do not contain access credentials.\n\n")

    lines.append("## Purpose\n\n")
    lines.append(
        "This log records the fixed per-package 80/10/10 episode split for the OpenVLA modified LIBERO RLDS dataset.\n\n"
    )
    lines.append("This task does not run OpenVLA inference and does not train any model.\n\n")
    lines.append(
        "This task only reads dataset_info.json and saves lightweight split metadata.\n\n"
    )

    lines.append("## Split Settings\n\n")
    lines.append(f"- Created at: `{split_data['created_at']}`\n")
    lines.append(f"- Dataset name: `{split_data['dataset_name']}`\n")
    lines.append(f"- Dataset root: `{split_data['dataset_root']}`\n")
    lines.append(f"- Source split: `{split_data['source_split']}`\n")
    lines.append(f"- Random seed: `{split_data['random_seed']}`\n")
    lines.append(f"- Split scope: `{split_data['split_scope']}`\n")
    lines.append(f"- Physical data copy: `{split_data['physical_data_copy']}`\n")
    lines.append(f"- Train ratio: `{ratios['train']}`\n")
    lines.append(f"- Validation ratio: `{ratios['val']}`\n")
    lines.append(f"- Test ratio: `{ratios['test']}`\n")
    lines.append(f"- Split JSON path: `{output_json}`\n")
    lines.append(f"- Split rule: {split_data['split_rule']}\n\n")

    lines.append("## Original Dataset Paths\n\n")
    lines.append("| Sub-dataset | Task suite | Version directory |\n")
    lines.append("|---|---|---|\n")

    for record in split_data["sub_datasets"]:
        lines.append(
            f"| {record['sub_dataset']} | {record['task_suite']} | `{record['version_dir']}` |\n"
        )

    lines.append("\n## Split Count Summary\n\n")
    lines.append("| Sub-dataset | Task suite | Original episodes | Train | Val | Test |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")

    for record in split_data["sub_datasets"]:
        lines.append(
            f"| {record['sub_dataset']} | {record['task_suite']} | "
            f"{record['original_episode_count']} | {record['train_count']} | "
            f"{record['val_count']} | {record['test_count']} |\n"
        )

    lines.append(
        f"| **Total** | **-** | **{totals['original_episode_count']}** | "
        f"**{totals['train_count']}** | **{totals['val_count']}** | "
        f"**{totals['test_count']}** |\n\n"
    )

    lines.append("## How This Split Will Be Used\n\n")
    lines.append(
        "The generated JSON file is the single source of truth for later dataset loading.\n\n"
    )
    lines.append(
        "The training loader should read `train_episode_indices` from each sub-dataset record.\n\n"
    )
    lines.append(
        "The validation loader should read `val_episode_indices` from each sub-dataset record.\n\n"
    )
    lines.append(
        "The test loader should read `test_episode_indices` only for final evaluation.\n\n"
    )
    lines.append(
        "For DF-03-02 baseline open-loop inference, a fixed 50-sample evaluation subset should be selected from the test split, not from the train or validation split.\n\n"
    )

    lines.append("## Important Git Rule\n\n")
    lines.append(
        "The generated JSON and Markdown log are lightweight metadata files and can be committed to GitHub.\n\n"
    )
    lines.append(
        "The real dataset files, TFRecord shards, model weights, cache folders, and credentials must not be committed.\n\n"
    )

    lines.append("## Interpretation\n\n")
    lines.append(
        "This split is a logical episode-level split by LIBERO sub-dataset. "
        "The original dataset folders remain unchanged.\n\n"
    )
    lines.append(
        "This means the project does not create physical train/val/test dataset copies. "
        "It only records which episode indices should be used for each split.\n"
    )

    output_log.write_text("".join(lines), encoding="utf-8")


# This function runs the full split creation flow and saves JSON plus Markdown outputs.
def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_json = Path(args.output_json)
    output_log = Path(args.output_log)

    print("=" * 80)
    print("DF-03-02 LIBERO Episode Split")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset root exists: {dataset_root.exists()}")
    print(f"Source split: {args.source_split}")
    print(f"Random seed: {args.seed}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Test ratio: {args.test_ratio}")
    print()

    split_data = create_all_splits(
        dataset_root=dataset_root,
        source_split=args.source_split,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    write_json(output_json=output_json, split_data=split_data)
    write_markdown_log(
        output_log=output_log,
        split_data=split_data,
        output_json=output_json,
    )

    print("Split count summary:")
    for record in split_data["sub_datasets"]:
        print(
            f"  {record['sub_dataset']}: "
            f"original={record['original_episode_count']}, "
            f"train={record['train_count']}, "
            f"val={record['val_count']}, "
            f"test={record['test_count']}"
        )

    print()
    print("Total:")
    print(
        f"  original={split_data['totals']['original_episode_count']}, "
        f"train={split_data['totals']['train_count']}, "
        f"val={split_data['totals']['val_count']}, "
        f"test={split_data['totals']['test_count']}"
    )

    print("=" * 80)
    print("Episode split finished.")
    print(f"JSON saved to: {output_json}")
    print(f"Markdown log saved to: {output_log}")
    print("=" * 80)


if __name__ == "__main__":
    main()