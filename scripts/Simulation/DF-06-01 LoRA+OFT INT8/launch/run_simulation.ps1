# Validate DF-06-01 locally without loading CUDA, OpenVLA, or LIBERO.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DefaultConfigPath = Join-Path $ProjectRoot "config\simulation_config.yaml"
$ConfigPath = if ($args.Count -gt 0) { $args[0] } else { $DefaultConfigPath }

# Confirm that every source file required by the project exists.
$RequiredFiles = @(
    "run_simulation.py",
    "requirements-int8.txt",
    "config\simulation_config.yaml",
    "src\__init__.py",
    "src\int8_quantization.py",
    "src\lora_oft_int8_policy.py",
    "src\libero_env.py",
    "src\simulation_runner.py",
    "diagnostics\int8_model_audit.py",
    "analysis\compare_bf16_int8.py",
    "launch\run_simulation.sh"
)

foreach ($RelativePath in $RequiredFiles) {
    $FullPath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "Missing required file: $FullPath"
    }
}

Push-Location $ProjectRoot
try {
    # Compile Python syntax without importing cloud-only dependencies.
    python -m py_compile `
        "run_simulation.py" `
        "src\int8_quantization.py" `
        "src\lora_oft_int8_policy.py" `
        "src\libero_env.py" `
        "src\simulation_runner.py" `
        "diagnostics\int8_model_audit.py" `
        "analysis\compare_bf16_int8.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Python syntax validation failed."
    }

    # Validate every fixed LoRA+OFT cross-field contract in the YAML file.
    python "run_simulation.py" --config $ConfigPath --validate-only
    if ($LASTEXITCODE -ne 0) {
        throw "Simulation config validation failed."
    }
}
finally {
    Pop-Location
}

Write-Host "DF-06-01 local validation: OK" -ForegroundColor Green
Write-Host "Run launch/run_simulation.sh in the openvla310 cloud environment."
