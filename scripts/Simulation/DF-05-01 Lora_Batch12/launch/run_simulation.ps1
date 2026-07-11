# Validate the DF-05-01 project locally before syncing it to the cloud.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $ProjectRoot "config\simulation_config.yaml"

# Confirm that every source file required by the project exists.
$RequiredFiles = @(
    "run_simulation.py",
    "config\simulation_config.yaml",
    "src\__init__.py",
    "src\lora_policy.py",
    "src\libero_env.py",
    "src\simulation_runner.py",
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
        "src\lora_policy.py" `
        "src\libero_env.py" `
        "src\simulation_runner.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Python syntax validation failed."
    }

    # Validate all YAML cross-field contracts.
    python "run_simulation.py" --config $ConfigPath --validate-only
    if ($LASTEXITCODE -ne 0) {
        throw "Simulation config validation failed."
    }
}
finally {
    Pop-Location
}

Write-Host "DF-05-01 local validation: OK" -ForegroundColor Green
Write-Host "Run launch/run_simulation.sh in the openvla310 cloud environment."
