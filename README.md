# OpenVLA Efficiency Project

This repository is for my MSc dissertation project on efficient deployment of OpenVLA for robotic manipulation.

## Project Aim

The main aim of this project is to improve the efficiency of OpenVLA-based robot control.

The project will focus on:

- LoRA-based fine-tuning
- QLoRA-based fine-tuning
- Quantized inference
- Latency and memory evaluation
- Lightweight world-state / tiny world model extension

## Important Environment Rule

This project uses:

```text
Python 3.10
```

All dependencies and code should be compatible with Python 3.10.

The main OpenVLA model will not mainly run on my local laptop. The model will run on:

```text
Cloud GPU / University HPC
```

The local laptop is mainly used for:

- Code editing
- Git management
- Light debugging
- Writing documentation

## Repository Structure

```text
configs/                         Configuration files
data/raw/                        Raw data placeholder, large data should not be committed
data/processed/                  Processed data placeholder
docs/                            Project documentation
experiments/01_baseline/         Baseline OpenVLA experiments
experiments/02_lora/             LoRA experiments
experiments/03_qlora/            QLoRA experiments
experiments/04_quantization/     Quantization experiments
experiments/05_world_model/      Lightweight world model experiments
results/tables/                  Result tables
results/figures/                 Result figures
results/logs/                    Log files, should not be committed if large
src/dataset/                     Dataset loading and preprocessing code
src/models/                      Model-related code
src/evaluation/                  Evaluation scripts
src/utils/                       Utility functions
```

## Git Branch Rule

Task branches should use the format:

```text
DF-number-number
```

Examples:

```text
DF-01-01
DF-01-02
DF-02-01
```

## Notes

Large files should not be committed to GitHub, including:

- Model checkpoints
- OpenVLA weights
- Raw datasets
- Large logs
- Large experiment outputs


