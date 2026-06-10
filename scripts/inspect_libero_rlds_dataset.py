# Notes:
# The following default paths are local paths on the my cloud machine.
# They are recorded for experiment reproducibility and do not contain access credentials.
#
# DF-03-02 LIBERO RLDS Dataset Inspection Script
#
# This script inspects the downloaded OpenVLA modified LIBERO RLDS dataset before
# the formal baseline evaluation runner is built.
#
# This script does not run OpenVLA inference.
# This script does not load OpenVLA model weights.
# This script does not train or fine-tune any model.
# This script only checks dataset metadata, TFRecord files, split names, episode counts,
# and one small preview episode from each LIBERO sub-dataset.
#
# Expected cloud dataset root:
# /root/autodl-tmp/datasets/openvla_modified_libero_rlds
#
# Expected outputs:
# logs/df_03_02_libero_dataset_inspection_log.md
# results/tables/df_03_02_libero_dataset_summary.csv

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# These four sub-dataset names are the confirmed folders in openvla/modified_libero_rlds.
DEFAULT_SUB_DATASETS: List[str] = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


# This function reads command-line settings so the script can run on the cloud server or another machine.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect OpenVLA modified LIBERO RLDS dataset structure."
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

    # The current dataset check uses train because the downloaded dataset only showed train episodes.
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to inspect. The current downloaded LIBERO dataset uses train split.",
    )

    # This flag enables a small real data preview through TensorFlow Datasets.
    parser.add_argument(
        "--read-first-episode",
        action="store_true",
        help="Read one episode from each sub-dataset to check real feature keys and tensor shapes.",
    )

    # This keeps the preview small so the inspection does not use too much memory.
    parser.add_argument(
        "--max-preview-steps",
        type=int,
        default=2,
        help="Limit the number of preview steps to keep memory use small.",
    )

    # This Markdown file records the human-readable inspection result.
    parser.add_argument(
        "--output-log",
        type=str,
        default="logs/df_03_02_libero_dataset_inspection_log.md",
        help="Path for the Markdown inspection log.",
    )

    # This CSV file records the machine-readable dataset summary.
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/tables/df_03_02_libero_dataset_summary.csv",
        help="Path for the CSV dataset summary.",
    )

    return parser.parse_args()


# This function loads dataset_info.json because it stores the TFDS metadata.
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# This function finds the version folder, such as 1.0.0, under each LIBERO sub-dataset folder.
def find_dataset_version_dir(sub_dataset_dir: Path) -> Optional[Path]:
    if not sub_dataset_dir.exists():
        return None

    # A valid TFDS version folder should contain dataset_info.json.
    for child in sorted(sub_dataset_dir.iterdir()):
        if child.is_dir() and (child / "dataset_info.json").exists():
            return child

    return None


# This function extracts available split names, such as train, val, or validation.
def extract_split_names(dataset_info: Dict[str, Any]) -> List[str]:
    splits = dataset_info.get("splits", [])

    # Some TFDS metadata stores splits as a list of dictionaries.
    if isinstance(splits, list):
        names: List[str] = []

        for split in splits:
            if isinstance(split, dict) and "name" in split:
                names.append(str(split["name"]))

        return names

    # Some TFDS metadata stores splits as a dictionary.
    if isinstance(splits, dict):
        return [str(name) for name in splits.keys()]

    return []


# This function extracts the episode count for the selected split from common TFDS metadata formats.
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

            if "shardLengths" in split and isinstance(split["shardLengths"], list):
                return int(sum(split["shardLengths"]))

            if "shard_lengths" in split and isinstance(split["shard_lengths"], list):
                return int(sum(split["shard_lengths"]))

    # This branch handles split metadata stored as a dictionary.
    if isinstance(splits, dict):
        split = splits.get(split_name, {})

        if isinstance(split, dict):
            if "numExamples" in split:
                return int(split["numExamples"])

            if "num_examples" in split:
                return int(split["num_examples"])

            if "shardLengths" in split and isinstance(split["shardLengths"], list):
                return int(sum(split["shardLengths"]))

            if "shard_lengths" in split and isinstance(split["shard_lengths"], list):
                return int(sum(split["shard_lengths"]))

    return 0


# This function summarizes tensors by shape and dtype instead of printing large arrays.
def summarize_tensor_or_value(value: Any) -> str:
    try:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)

        if shape is not None or dtype is not None:
            return f"shape={shape}, dtype={dtype}"

        return f"type={type(value).__name__}"

    except Exception as error:
        return f"summary_error={error}"


# This function collects nested dictionary keys so the log can show where important fields may live.
def summarize_nested_keys(
    obj: Any,
    max_depth: int = 2,
    current_depth: int = 0,
) -> List[str]:
    if current_depth > max_depth:
        return []

    keys: List[str] = []

    # TensorFlow dataset examples usually behave like nested dictionaries.
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = str(key)
            keys.append(key_text)

            # Nested dictionaries are expanded into dotted key names.
            if isinstance(value, dict):
                nested_keys = summarize_nested_keys(
                    value,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )

                for nested_key in nested_keys:
                    keys.append(f"{key_text}.{nested_key}")

    return keys


# This function reads one episode from a local TFDS folder to inspect real episode and step fields.
def try_read_first_episode(
    dataset_dir: Path,
    split_name: str,
    max_preview_steps: int,
) -> Dict[str, Any]:
    preview: Dict[str, Any] = {
        "read_success": False,
        "episode_keys": "",
        "step_keys": "",
        "observation_keys": "",
        "action_summary": "",
        "language_candidates": "",
        "error": "",
    }

    # tensorflow_datasets is imported inside the function because local VS Code may not have it.
    try:
        import tensorflow_datasets as tfds
    except Exception as error:
        preview["error"] = f"tensorflow_datasets import failed: {error}"
        return preview

    try:
        # builder_from_directory loads the already-downloaded TFDS dataset from disk.
        builder = tfds.builder_from_directory(str(dataset_dir))

        # shuffle_files is disabled so repeated inspection gives stable preview results.
        dataset = builder.as_dataset(split=split_name, shuffle_files=False)

        # Only one episode is read because DF-03-02 is an inspection task.
        first_episode = next(iter(dataset.take(1)))

        # Episode keys show the high-level RLDS structure.
        preview["episode_keys"] = ", ".join(
            summarize_nested_keys(first_episode, max_depth=1)
        )

        if "steps" not in first_episode:
            preview["error"] = "No 'steps' field found in the first episode."
            return preview

        steps = first_episode["steps"]
        first_step = None

        # Only the first readable step is needed to inspect field names and tensor shapes.
        for _, step in zip(range(max_preview_steps), steps):
            first_step = step
            break

        if first_step is None:
            preview["error"] = "The first episode contains no readable steps."
            return preview

        # Step keys show where observation, action, reward, and instruction fields may be.
        preview["step_keys"] = ", ".join(
            summarize_nested_keys(first_step, max_depth=1)
        )

        # Observation keys help identify the image field needed by OpenVLA.
        if isinstance(first_step, dict) and "observation" in first_step:
            observation = first_step["observation"]

            if isinstance(observation, dict):
                preview["observation_keys"] = ", ".join(
                    str(key) for key in observation.keys()
                )

        # Action shape is checked before building action parsing or comparison logic.
        if isinstance(first_step, dict) and "action" in first_step:
            preview["action_summary"] = summarize_tensor_or_value(first_step["action"])

        # These are common language field names used by robot learning datasets.
        possible_language_keys = [
            "language_instruction",
            "natural_language_instruction",
            "instruction",
            "language",
        ]

        found_language_fields: List[str] = []

        # Language fields may appear directly under step.
        if isinstance(first_step, dict):
            for key in possible_language_keys:
                if key in first_step:
                    found_language_fields.append(
                        f"step.{key}: {summarize_tensor_or_value(first_step[key])}"
                    )

            # Language fields may also appear inside observation.
            if "observation" in first_step and isinstance(first_step["observation"], dict):
                for key in possible_language_keys:
                    if key in first_step["observation"]:
                        found_language_fields.append(
                            "step.observation."
                            f"{key}: {summarize_tensor_or_value(first_step['observation'][key])}"
                        )

        preview["language_candidates"] = "; ".join(found_language_fields)
        preview["read_success"] = True

        return preview

    except Exception as error:
        preview["error"] = str(error)
        return preview


# This function inspects one LIBERO sub-dataset and returns one summary row.
def inspect_sub_dataset(
    dataset_root: Path,
    sub_dataset_name: str,
    split_name: str,
    read_first_episode: bool,
    max_preview_steps: int,
) -> Dict[str, Any]:
    sub_dataset_dir = dataset_root / sub_dataset_name
    version_dir = find_dataset_version_dir(sub_dataset_dir)

    # This row is written directly into the final CSV file.
    row: Dict[str, Any] = {
        "sub_dataset": sub_dataset_name,
        "sub_dataset_dir": str(sub_dataset_dir),
        "version_dir": str(version_dir) if version_dir else "",
        "exists": sub_dataset_dir.exists(),
        "dataset_info_exists": False,
        "split_names": "",
        "train_episodes": 0,
        "tfrecord_files": 0,
        "incomplete_files": 0,
        "read_first_episode_success": False,
        "episode_keys": "",
        "step_keys": "",
        "observation_keys": "",
        "action_summary": "",
        "language_candidates": "",
        "error": "",
    }

    if version_dir is None:
        row["error"] = "Version directory with dataset_info.json was not found."
        return row

    dataset_info_path = version_dir / "dataset_info.json"
    row["dataset_info_exists"] = dataset_info_path.exists()

    # dataset_info.json provides split names and episode counts.
    try:
        dataset_info = load_json(dataset_info_path)
        split_names = extract_split_names(dataset_info)
        split_count = extract_split_count(dataset_info, split_name)

        row["split_names"] = ",".join(split_names)
        row["train_episodes"] = split_count

    except Exception as error:
        row["error"] = f"Failed to read dataset_info.json: {error}"

    # TFRecord files are the actual dataset shard files.
    tfrecord_files = list(version_dir.rglob("*.tfrecord*"))

    # Incomplete files indicate broken or unfinished downloads.
    incomplete_files = list(version_dir.rglob("*.incomplete"))

    row["tfrecord_files"] = len(tfrecord_files)
    row["incomplete_files"] = len(incomplete_files)

    # The first episode preview confirms the real field structure.
    if read_first_episode:
        preview = try_read_first_episode(
            dataset_dir=version_dir,
            split_name=split_name,
            max_preview_steps=max_preview_steps,
        )

        row["read_first_episode_success"] = preview["read_success"]
        row["episode_keys"] = preview["episode_keys"]
        row["step_keys"] = preview["step_keys"]
        row["observation_keys"] = preview["observation_keys"]
        row["action_summary"] = preview["action_summary"]
        row["language_candidates"] = preview["language_candidates"]

        if preview["error"]:
            row["error"] = preview["error"]

    return row


# This function writes a small metadata-only CSV that is safe to commit to GitHub.
def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sub_dataset",
        "sub_dataset_dir",
        "version_dir",
        "exists",
        "dataset_info_exists",
        "split_names",
        "train_episodes",
        "tfrecord_files",
        "incomplete_files",
        "read_first_episode_success",
        "episode_keys",
        "step_keys",
        "observation_keys",
        "action_summary",
        "language_candidates",
        "error",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# This function writes the Markdown log used for project tracking and dissertation notes.
def write_markdown_log(
    rows: List[Dict[str, Any]],
    output_log: Path,
    dataset_root: Path,
    split_name: str,
) -> None:
    output_log.parent.mkdir(parents=True, exist_ok=True)

    total_episodes = sum(int(row.get("train_episodes", 0)) for row in rows)
    total_tfrecords = sum(int(row.get("tfrecord_files", 0)) for row in rows)
    total_incomplete = sum(int(row.get("incomplete_files", 0)) for row in rows)

    lines: List[str] = []

    lines.append("# DF-03-02 LIBERO Dataset Inspection Log\n\n")

    lines.append("## Notes\n\n")
    lines.append("The following paths are local paths on the author's cloud machine.\n")
    lines.append("They are recorded for experiment reproducibility and do not contain access credentials.\n\n")

    lines.append("## Purpose\n\n")
    lines.append(
        "This log records the structure inspection of the OpenVLA modified LIBERO RLDS dataset.\n\n"
    )
    lines.append(
        "This task does not run OpenVLA inference and does not train any model.\n\n"
    )

    lines.append("## Inspection Environment\n\n")
    lines.append(f"- Inspection time: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"- Python executable: `{sys.executable}`\n")
    lines.append(f"- Dataset root: `{dataset_root}`\n")
    lines.append(f"- Split inspected: `{split_name}`\n\n")

    lines.append("## Summary\n\n")
    lines.append(f"- Total sub-datasets inspected: {len(rows)}\n")
    lines.append(f"- Total train episodes: {total_episodes}\n")
    lines.append(f"- Total TFRecord files: {total_tfrecords}\n")
    lines.append(f"- Total incomplete files: {total_incomplete}\n\n")

    lines.append("## Sub-dataset Summary\n\n")
    lines.append(
        "| Sub-dataset | Split names | Train episodes | TFRecord files | "
        "Incomplete files | First episode read |\n"
    )
    lines.append("|---|---|---:|---:|---:|---|\n")

    for row in rows:
        lines.append(
            f"| {row['sub_dataset']} | {row['split_names']} | {row['train_episodes']} | "
            f"{row['tfrecord_files']} | {row['incomplete_files']} | "
            f"{row['read_first_episode_success']} |\n"
        )

    lines.append("\n## First Episode Preview\n\n")

    for row in rows:
        lines.append(f"### {row['sub_dataset']}\n\n")
        lines.append(f"- Version dir: `{row['version_dir']}`\n")
        lines.append(f"- Episode keys: `{row['episode_keys']}`\n")
        lines.append(f"- Step keys: `{row['step_keys']}`\n")
        lines.append(f"- Observation keys: `{row['observation_keys']}`\n")
        lines.append(f"- Action summary: `{row['action_summary']}`\n")
        lines.append(f"- Language candidates: `{row['language_candidates']}`\n")
        lines.append(f"- Error: `{row['error']}`\n\n")

    lines.append("## Interpretation\n\n")
    lines.append(
        "The inspection confirms the available dataset fields before building the formal baseline runner.\n\n"
    )
    lines.append(
        "The next step is to build a sample loader that converts image observations and "
        "language instructions into OpenVLA-compatible inputs.\n\n"
    )

    if total_incomplete == 0:
        lines.append("Download completeness status: OK. No incomplete files were found.\n")
    else:
        lines.append(
            "Download completeness status: WARNING. Incomplete files were found and should be checked.\n"
        )

    output_log.write_text("".join(lines), encoding="utf-8")


# This function runs the full inspection flow and saves both CSV and Markdown outputs.
def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_log = Path(args.output_log)
    output_csv = Path(args.output_csv)

    rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("DF-03-02 LIBERO Dataset Inspection")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset root exists: {dataset_root.exists()}")
    print(f"Split: {args.split}")
    print(f"Read first episode: {args.read_first_episode}")
    print()

    for sub_dataset_name in DEFAULT_SUB_DATASETS:
        print(f"Inspecting: {sub_dataset_name}")

        row = inspect_sub_dataset(
            dataset_root=dataset_root,
            sub_dataset_name=sub_dataset_name,
            split_name=args.split,
            read_first_episode=args.read_first_episode,
            max_preview_steps=args.max_preview_steps,
        )

        rows.append(row)

        print(f"  version_dir: {row['version_dir']}")
        print(f"  split_names: {row['split_names']}")
        print(f"  train_episodes: {row['train_episodes']}")
        print(f"  tfrecord_files: {row['tfrecord_files']}")
        print(f"  incomplete_files: {row['incomplete_files']}")
        print(f"  first_episode_read: {row['read_first_episode_success']}")

        if row["error"]:
            print(f"  error: {row['error']}")

        print()

    write_csv(rows=rows, output_csv=output_csv)
    write_markdown_log(
        rows=rows,
        output_log=output_log,
        dataset_root=dataset_root,
        split_name=args.split,
    )

    print("=" * 80)
    print("Inspection finished.")
    print(f"CSV saved to: {output_csv}")
    print(f"Markdown log saved to: {output_log}")
    print("=" * 80)


if __name__ == "__main__":
    main()