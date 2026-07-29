# Build and audit the mixed-precision INT8 inference configuration.

import importlib.metadata
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn
from transformers import BitsAndBytesConfig


# Return an installed package version or None when metadata is unavailable.
def package_version(package_name: str) -> Optional[str]:
    # Keep missing optional packages explicit in the runtime audit.
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Import bitsandbytes and expose a clear installation failure.
def import_bitsandbytes() -> Any:
    # Delay the import until cloud inference so local static validation still works.
    try:
        import bitsandbytes as bnb
    except Exception as error:
        raise RuntimeError(
            "bitsandbytes could not be imported in the simulation environment. "
            "Install a CUDA-12.8 and RTX-5090 compatible bitsandbytes build."
        ) from error
    return bnb


# Construct the exact Hugging Face LLM.int8 configuration from YAML.
def build_int8_quantization_config(
    quantization_config: Mapping[str, Any],
) -> BitsAndBytesConfig:
    # Reject alternate methods so this experiment cannot silently become INT4.
    if not bool(quantization_config["enabled"]):
        raise ValueError("INT8 quantization must remain enabled.")
    if str(quantization_config["backend"]) != "bitsandbytes":
        raise ValueError("quantization.backend must be 'bitsandbytes'.")
    if str(quantization_config["method"]) != "llm_int8":
        raise ValueError("quantization.method must be 'llm_int8'.")
    if not bool(quantization_config["load_in_8bit"]):
        raise ValueError("quantization.load_in_8bit must be true.")
    if bool(quantization_config["load_in_4bit"]):
        raise ValueError("INT4 loading must remain disabled in DF-06-01.")

    skip_modules = [
        str(module_name)
        for module_name in quantization_config["skip_modules"]
    ]
    if not skip_modules:
        raise ValueError("quantization.skip_modules cannot be empty.")

    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=float(
            quantization_config["llm_int8_threshold"]
        ),
        llm_int8_skip_modules=skip_modules,
        llm_int8_enable_fp32_cpu_offload=bool(
            quantization_config["llm_int8_enable_fp32_cpu_offload"]
        ),
    )


# Cast only trainable LoRA adapter parameters without moving the quantized base.
def cast_lora_parameters(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> List[str]:
    # A whole-model .to call is invalid after bitsandbytes quantization.
    converted_names: List[str] = []
    for name, parameter in model.named_parameters():
        if "lora_" not in name.lower():
            continue
        parameter.data = parameter.data.to(device=device, dtype=dtype)
        converted_names.append(name)
    if not converted_names:
        raise RuntimeError("No LoRA parameters were found after adapter loading.")
    return converted_names


# Return sorted floating-point dtypes for parameters selected by name.
def parameter_dtypes(
    model: nn.Module,
    name_fragments: Sequence[str],
) -> List[str]:
    # Exclude integer quantized weights because this helper audits BF16 components.
    dtypes = {
        str(parameter.dtype).replace("torch.", "")
        for name, parameter in model.named_parameters()
        if any(fragment in name for fragment in name_fragments)
        and parameter.is_floating_point()
    }
    return sorted(dtypes)


# Return a JSON-safe count of parameter devices.
def parameter_device_counts(model: nn.Module) -> Dict[str, int]:
    # Count parameter tensors rather than bytes to expose any CPU-resident shard.
    counts: Dict[str, int] = {}
    for parameter in model.parameters():
        device_name = str(parameter.device)
        counts[device_name] = counts.get(device_name, 0) + 1
    return dict(sorted(counts.items()))


# Find the device map created by Transformers model dispatch.
def find_hf_device_map(model: nn.Module) -> Dict[str, str]:
    # PEFT may expose the map on either the wrapper or its base model.
    candidates = [model]
    if hasattr(model, "get_base_model"):
        candidates.append(model.get_base_model())
    for candidate in candidates:
        raw_map = getattr(candidate, "hf_device_map", None)
        if isinstance(raw_map, dict):
            return {
                str(name): str(device)
                for name, device in sorted(raw_map.items())
            }
    return {}


# Find the Transformers flag proving that 8-bit loading was requested.
def find_loaded_in_8bit(model: nn.Module) -> bool:
    # Check both PEFT and base wrappers because version-specific placement varies.
    candidates = [model]
    if hasattr(model, "get_base_model"):
        candidates.append(model.get_base_model())
    for candidate in candidates:
        if bool(getattr(candidate, "is_loaded_in_8bit", False)):
            return True
    return False


# Estimate the in-memory model footprint in GiB.
def model_memory_footprint_gib(model: nn.Module) -> float:
    # Prefer the Transformers implementation because it understands quantized tensors.
    getter = getattr(model, "get_memory_footprint", None)
    if callable(getter):
        return float(getter()) / 1024**3
    total_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(model.parameters()) + list(model.buffers())
    )
    return float(total_bytes) / 1024**3


# Return the floating-point dtype set for one standalone component.
def standalone_component_dtypes(component: nn.Module) -> List[str]:
    # Report only floating tensors because integer counters are not compute weights.
    return sorted(
        {
            str(parameter.dtype).replace("torch.", "")
            for parameter in component.parameters()
            if parameter.is_floating_point()
        }
    )


# Build a complete, JSON-safe audit of the loaded mixed-precision model.
def build_quantization_audit(
    model: nn.Module,
    action_head: nn.Module,
    proprio_projector: nn.Module,
    quantization_config: Mapping[str, Any],
    lora_parameter_names: Sequence[str],
    model_load_seconds: float,
    load_peak_vram_allocated_gib: float,
    load_peak_vram_reserved_gib: float,
) -> Dict[str, Any]:
    # Count actual bitsandbytes modules instead of trusting configuration metadata.
    bnb = import_bitsandbytes()
    quantized_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear8bitLt)
    ]
    torch_linear_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not isinstance(module, bnb.nn.Linear8bitLt)
    ]
    device_map = find_hf_device_map(model)
    cpu_or_disk_entries = {
        name: device
        for name, device in device_map.items()
        if device.lower() in {"cpu", "disk"}
    }

    # PEFT names include wrapper prefixes, so substring matching is deliberate.
    component_dtypes = {
        "vision_backbone": parameter_dtypes(
            model,
            ("vision_backbone",),
        ),
        "multimodal_projector": parameter_dtypes(
            model,
            (".projector.",),
        ),
        "language_model_lm_head": parameter_dtypes(
            model,
            (".lm_head.",),
        ),
        "lora_adapters": parameter_dtypes(
            model,
            ("lora_",),
        ),
        "action_head": standalone_component_dtypes(action_head),
        "proprio_projector": standalone_component_dtypes(
            proprio_projector
        ),
    }

    return {
        "enabled": bool(quantization_config["enabled"]),
        "backend": str(quantization_config["backend"]),
        "method": str(quantization_config["method"]),
        "load_in_8bit_requested": bool(
            quantization_config["load_in_8bit"]
        ),
        "load_in_4bit_requested": bool(
            quantization_config["load_in_4bit"]
        ),
        "is_loaded_in_8bit": find_loaded_in_8bit(model),
        "llm_int8_threshold": float(
            quantization_config["llm_int8_threshold"]
        ),
        "llm_int8_skip_modules": [
            str(value) for value in quantization_config["skip_modules"]
        ],
        "llm_int8_enable_fp32_cpu_offload": bool(
            quantization_config["llm_int8_enable_fp32_cpu_offload"]
        ),
        "quantized_linear_module_count": len(quantized_modules),
        "quantized_linear_module_names": sorted(quantized_modules),
        "remaining_torch_linear_module_count": len(torch_linear_modules),
        "remaining_torch_linear_module_names": sorted(
            torch_linear_modules
        ),
        "lora_parameter_count": len(lora_parameter_names),
        "lora_parameter_names": sorted(lora_parameter_names),
        "component_dtypes": component_dtypes,
        "parameter_device_counts": parameter_device_counts(model),
        "hf_device_map": device_map,
        "cpu_or_disk_device_map_entries": cpu_or_disk_entries,
        "model_memory_footprint_gib": model_memory_footprint_gib(model),
        "model_load_seconds": float(model_load_seconds),
        "load_peak_vram_allocated_gib": float(
            load_peak_vram_allocated_gib
        ),
        "load_peak_vram_reserved_gib": float(
            load_peak_vram_reserved_gib
        ),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "accelerate_version": package_version("accelerate"),
    }


# Enforce that the loaded model matches the declared mixed-precision contract.
def validate_quantization_audit(
    audit: Mapping[str, Any],
    quantization_config: Mapping[str, Any],
) -> None:
    # Fail before simulation when INT8 was requested but not actually materialized.
    failures: List[str] = []
    if not bool(audit["is_loaded_in_8bit"]):
        failures.append("Transformers is_loaded_in_8bit is false")
    if int(audit["quantized_linear_module_count"]) <= 0:
        failures.append("no bitsandbytes Linear8bitLt modules were found")

    quantized_names = [
        str(name) for name in audit["quantized_linear_module_names"]
    ]
    outside_language_model = [
        name for name in quantized_names if "language_model" not in name
    ]
    if outside_language_model:
        failures.append(
            "INT8 modules exist outside language_model: "
            + ", ".join(outside_language_model[:5])
        )

    skip_modules = [
        str(name) for name in quantization_config["skip_modules"]
    ]
    violated_skips = [
        name
        for name in quantized_names
        if any(skip_name in name for skip_name in skip_modules)
    ]
    if violated_skips:
        failures.append(
            "skipped modules were quantized: "
            + ", ".join(violated_skips[:5])
        )

    if audit["cpu_or_disk_device_map_entries"]:
        failures.append("the Hugging Face device map contains CPU or disk offload")
    non_cuda_parameter_devices = [
        device
        for device in audit["parameter_device_counts"]
        if not str(device).startswith("cuda")
    ]
    if (
        bool(quantization_config["forbid_cpu_or_disk_offload"])
        and non_cuda_parameter_devices
    ):
        failures.append(
            "model parameters remain outside CUDA: "
            + ", ".join(non_cuda_parameter_devices)
        )

    expected_dtype = "bfloat16"
    required_bfloat16_components = (
        "vision_backbone",
        "multimodal_projector",
        "language_model_lm_head",
        "lora_adapters",
        "action_head",
        "proprio_projector",
    )
    for component_name in required_bfloat16_components:
        dtypes = list(audit["component_dtypes"][component_name])
        if dtypes != [expected_dtype]:
            failures.append(
                f"{component_name} dtypes are {dtypes}, expected ['bfloat16']"
            )

    if int(audit["lora_parameter_count"]) <= 0:
        failures.append("no LoRA adapter parameters were audited")
    if failures:
        raise RuntimeError("INT8 quantization audit failed: " + "; ".join(failures))
