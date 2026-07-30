# Load the DF-06-02 policy and write a standalone NF4 materialization audit.

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


# Add the project root before importing the local simulation entry point.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_simulation import (  # noqa: E402
    configure_logging,
    configure_runtime_environment,
    load_config,
    resolve_output_paths,
    seed_everything,
    validate_config,
    validate_runtime_identity,
)


# Parse the simulation config and optional report path.
def parse_args() -> argparse.Namespace:
    # Keep the default paths aligned with the main launcher.
    parser = argparse.ArgumentParser(
        description="Audit the loaded DF-06-02 bitsandbytes NF4 model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "simulation_config.yaml",
        help="Path to the DF-06-02 simulation YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


# Write one JSON mapping atomically.
def write_json(path: Path, document: Mapping[str, Any]) -> None:
    # Avoid leaving a partial audit when the process is interrupted.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)


# Load the model, enforce every hard audit, and save the proof record.
def main() -> int:
    # Reuse the formal configuration validator so audit and rollout cannot diverge.
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    resolve_output_paths(config)
    configure_runtime_environment(config)
    configure_logging(config)
    seed_everything(int(config["experiment"]["seed"]))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The NF4 model audit requires CUDA.")
    validate_runtime_identity(config)

    # Import after source paths are fixed to the DF-04-03 implementation.
    from src.lora_oft_int4_policy import LoraOFTINT4Policy

    policy = LoraOFTINT4Policy(config)
    output_path = args.output
    if output_path is None:
        output_path = (
            Path(config["paths"]["reports_directory"])
            / "int4_model_audit.json"
        )
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    write_json(output_path, policy.quantization_audit)

    print("DF-06-02 NF4 model audit: OK")
    print(
        "quantized_linear_module_count:",
        policy.quantization_audit["quantized_linear_module_count"],
    )
    print(
        "model_memory_footprint_gib:",
        policy.quantization_audit["model_memory_footprint_gib"],
    )
    print("report:", output_path)
    return 0


# Exit with the diagnostic return code.
if __name__ == "__main__":
    raise SystemExit(main())
