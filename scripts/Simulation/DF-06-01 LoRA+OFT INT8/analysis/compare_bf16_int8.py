# Compare DF-05-02 BF16 and DF-06-01 INT8 structured simulation results.

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


# Resolve the project root from this analysis script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Define the comparable aggregate fields retained by both experiments.
COMPARISON_FIELDS = (
    "overall_success_rate",
    "macro_average_task_success_rate",
    "overall_timeout_rate",
    "avg_success_steps",
    "avg_all_steps",
    "avg_inference_latency_ms",
    "p95_inference_latency_ms",
    "avg_amortized_inference_ms_per_action",
    "avg_end_to_end_latency_ms",
    "avg_amortized_end_to_end_ms_per_action",
    "avg_control_frequency_hz",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
    "zero_equivalent_arm_action_rate",
    "zero_equivalent_full_noop_rate",
    "repeated_action_count",
    "repeated_chunk_count",
    "collapse_warning_count",
)


# Parse the BF16 baseline, INT8 result, and output locations.
def parse_args() -> argparse.Namespace:
    # Require the baseline path because cloud and local layouts can differ.
    parser = argparse.ArgumentParser(
        description="Compare DF-05-02 BF16 with DF-06-01 INT8."
    )
    parser.add_argument(
        "--bf16-metrics",
        type=Path,
        required=True,
        help="DF-05-02 output/metrics directory.",
    )
    parser.add_argument(
        "--int8-metrics",
        type=Path,
        default=PROJECT_ROOT / "output" / "metrics",
        help="DF-06-01 output/metrics directory.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "output" / "reports",
        help="Directory for JSON, CSV, and Markdown comparison reports.",
    )
    return parser.parse_args()


# Load one JSON object from disk.
def load_json(path: Path) -> Dict[str, Any]:
    # Reject missing or non-object records before computing deltas.
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return document


# Load all rows from one task-summary CSV.
def load_csv(path: Path) -> List[Dict[str, str]]:
    # Preserve raw strings so exact source values remain traceable.
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# Convert a nullable value to float.
def optional_float(value: Any) -> Optional[float]:
    # Preserve None and empty strings for unavailable metrics.
    if value is None or str(value).strip() == "":
        return None
    return float(value)


# Build one numeric comparison row with absolute and relative deltas.
def compare_value(
    field: str,
    baseline_value: Any,
    quantized_value: Any,
) -> Dict[str, Any]:
    # Use INT8 minus BF16 so negative latency and VRAM deltas mean reductions.
    baseline = optional_float(baseline_value)
    quantized = optional_float(quantized_value)
    if baseline is None or quantized is None:
        absolute_delta = None
        relative_delta = None
    else:
        absolute_delta = quantized - baseline
        relative_delta = (
            100.0 * absolute_delta / baseline
            if baseline != 0.0
            else None
        )
    return {
        "metric": field,
        "bf16": baseline,
        "int8": quantized,
        "int8_minus_bf16": absolute_delta,
        "relative_change_percent": relative_delta,
    }


# Validate that both result sets describe the same checkpoint and protocol size.
def validate_comparability(
    baseline: Mapping[str, Any],
    quantized: Mapping[str, Any],
) -> None:
    # Fail on mismatched checkpoint or rollout counts rather than report invalid deltas.
    failures: List[str] = []
    if int(baseline["checkpoint_step"]) != int(quantized["checkpoint_step"]):
        failures.append("checkpoint_step differs")
    if int(baseline["total_tasks"]) != int(quantized["total_tasks"]):
        failures.append("total_tasks differs")
    if int(baseline["total_rollouts"]) != int(quantized["total_rollouts"]):
        failures.append("total_rollouts differs")
    if bool(baseline["completed"]) != bool(quantized["completed"]):
        failures.append("completed status differs")
    if failures:
        raise ValueError("Results are not directly comparable: " + ", ".join(failures))


# Compare per-task success rates by task index.
def compare_tasks(
    baseline_rows: Sequence[Mapping[str, str]],
    quantized_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, Any]]:
    # Join task rows by official LIBERO task index.
    baseline_by_task = {
        int(row["task_index"]): row for row in baseline_rows
    }
    quantized_by_task = {
        int(row["task_index"]): row for row in quantized_rows
    }
    if set(baseline_by_task) != set(quantized_by_task):
        raise ValueError("BF16 and INT8 task-summary indices differ.")

    comparisons: List[Dict[str, Any]] = []
    for task_index in sorted(baseline_by_task):
        baseline_row = baseline_by_task[task_index]
        quantized_row = quantized_by_task[task_index]
        comparison = compare_value(
            "success_rate",
            baseline_row["success_rate"],
            quantized_row["success_rate"],
        )
        comparisons.append(
            {
                "task_index": task_index,
                "task_name": quantized_row["task_name"],
                "bf16_success_rate": comparison["bf16"],
                "int8_success_rate": comparison["int8"],
                "int8_minus_bf16": comparison["int8_minus_bf16"],
            }
        )
    return comparisons


# Write one JSON document atomically.
def write_json(path: Path, document: Mapping[str, Any]) -> None:
    # Replace the target only after a complete temporary write.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)


# Write aggregate comparison rows to CSV.
def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    # Use one stable schema that is easy to import into thesis tables.
    fieldnames = [
        "metric",
        "bf16",
        "int8",
        "int8_minus_bf16",
        "relative_change_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Write a concise human-readable Markdown comparison.
def write_markdown(
    path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
) -> None:
    # Keep the report factual and derive every number from structured files.
    lines = [
        "# DF-05-02 BF16 vs DF-06-01 INT8",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | BF16 | INT8 | INT8 - BF16 | Relative change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        relative = row["relative_change_percent"]
        relative_text = "" if relative is None else f"{relative:.4f}%"
        lines.append(
            "| {metric} | {bf16} | {int8} | {delta} | {relative} |".format(
                metric=row["metric"],
                bf16=row["bf16"],
                int8=row["int8"],
                delta=row["int8_minus_bf16"],
                relative=relative_text,
            )
        )
    lines.extend(
        [
            "",
            "## Per-task success rate",
            "",
            "| Task | BF16 | INT8 | INT8 - BF16 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in task_rows:
        lines.append(
            f"| {row['task_index']} | {row['bf16_success_rate']} | "
            f"{row['int8_success_rate']} | {row['int8_minus_bf16']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Build and save all BF16-versus-INT8 comparison artifacts.
def main() -> int:
    # Read only finalized structured records, never parse console logs.
    args = parse_args()
    bf16_metrics = args.bf16_metrics.expanduser().resolve()
    int8_metrics = args.int8_metrics.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    baseline = load_json(bf16_metrics / "simulation_summary.json")
    quantized = load_json(int8_metrics / "simulation_summary.json")
    validate_comparability(baseline, quantized)

    aggregate_rows = [
        compare_value(field, baseline.get(field), quantized.get(field))
        for field in COMPARISON_FIELDS
    ]
    task_rows = compare_tasks(
        load_csv(bf16_metrics / "task_summary.csv"),
        load_csv(int8_metrics / "task_summary.csv"),
    )
    report = {
        "comparison": "DF-05-02 BF16 versus DF-06-01 INT8",
        "bf16_run_id": baseline["run_id"],
        "int8_run_id": quantized["run_id"],
        "checkpoint_step": quantized["checkpoint_step"],
        "total_rollouts": quantized["total_rollouts"],
        "aggregate_metrics": aggregate_rows,
        "per_task_success_rate": task_rows,
    }

    write_json(output_directory / "bf16_vs_int8_comparison.json", report)
    write_csv(output_directory / "bf16_vs_int8_comparison.csv", aggregate_rows)
    write_markdown(
        output_directory / "bf16_vs_int8_comparison.md",
        aggregate_rows,
        task_rows,
    )
    print("DF-05-02 BF16 versus DF-06-01 INT8 comparison: OK")
    print("reports:", output_directory)
    return 0


# Exit with the analysis return code.
if __name__ == "__main__":
    raise SystemExit(main())
