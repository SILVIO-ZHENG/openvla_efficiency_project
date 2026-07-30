# Validate DF-06-02 locally without loading CUDA, OpenVLA, or LIBERO.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DefaultConfigPath = Join-Path $ProjectRoot "config\simulation_config.yaml"
$ConfigPath = if ($args.Count -gt 0) { $args[0] } else { $DefaultConfigPath }

# Confirm that every source file required by the project exists.
$RequiredFiles = @(
    "run_simulation.py",
    "requirements-int4.txt",
    "config\simulation_config.yaml",
    "src\__init__.py",
    "src\int4_quantization.py",
    "src\lora_oft_int4_policy.py",
    "src\libero_env.py",
    "src\simulation_runner.py",
    "diagnostics\int4_model_audit.py",
    "analysis\compare_bf16_int4.py",
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
        "src\int4_quantization.py" `
        "src\lora_oft_int4_policy.py" `
        "src\libero_env.py" `
        "src\simulation_runner.py" `
        "diagnostics\int4_model_audit.py" `
        "analysis\compare_bf16_int4.py"
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

Write-Host "DF-06-02 local validation: OK" -ForegroundColor Green
Write-Host "Run launch/run_simulation.sh in the openvla310 cloud environment."
