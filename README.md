# OpenVLA Efficiency Project

This repository contains my experiments on efficient
fine-tuning and deployment of OpenVLA for robotic manipulation.

## Completed research scope

- OpenVLA LoRA fine-tuning
- OpenVLA LoRA+OFT continuous-action fine-tuning
- Closed-loop LIBERO-10 simulation
- BF16 inference evaluation
- Mixed-precision INT8 inference evaluation
- Mixed-precision INT4 inference evaluation
- Latency, control-frequency, VRAM, and action-collapse recording
- Quantization and deployment comparisons
- QLoRA evaluation
- Lightweight world-state extension

Project status:

```text
Completed
```

## Environment

All project code targets:

```text
Python 3.10
```

| Workload | Main environment |
|---|---|
| Code editing and Git | Local Windows workstation |
| Model training | Cloud GPU / university HPC |
| Closed-loop simulation | NVIDIA GeForce RTX 5090, 32 GB |
| Simulator | LIBERO-10 |

Large OpenVLA weights and datasets are stored outside Git.

## Repository structure

```text
scripts/
├── Training/                   DF-coded training projects
└── Simulation/
    ├── DF-05-01 Lora_Batch12/ LoRA closed-loop evaluation and diagnostics
    ├── DF-05-02 Lora+OFT/     LoRA+OFT BF16 evaluation
    └── DF-06-01 LoRA+OFT INT8/
                                LoRA+OFT mixed-precision INT8 evaluation

docs/                           Dissertation and project documentation
data/                           Local dataset placeholders and metadata
results/                        Cross-experiment tables and figures
```

Each DF project keeps its own configuration, launch scripts, source code,
README, logs, metrics, reports, and ignored video directory.

## Experiment map

| ID | Experiment | Status |
|---|---|---|
| DF-04-02 | LoRA training | Completed |
| DF-04-03 | LoRA+OFT continuous-action training | Completed |
| DF-05-01 | LoRA LIBERO-10 simulation and offline diagnostics | Completed |
| DF-05-02 | LoRA+OFT BF16 LIBERO-10 simulation | Completed |
| DF-06-01 | LoRA+OFT INT8 LIBERO-10 simulation | Completed |
| DF-06-02 | LoRA+OFT INT4 LIBERO-10 simulation | Completed |

## Recorded closed-loop results

Both rows use the DF-04-03 best adapter at step 100,000, the same LIBERO-10
task protocol, 50 trials per task, and 500 rollouts.

| Experiment | Precision | Success rate | Avg. inference | Control rate | Peak allocated VRAM |
|---|---|---:|---:|---:|---:|
| DF-05-02 | BF16 | 94.0% | 135.889 ms | 26.808 Hz | 15.193 GiB |
| DF-06-01 | INT8 language model + BF16 remaining components | 94.8% | 199.351 ms | 22.868 Hz | 9.263 GiB |

DF-06-01 quantizes 224 eligible language-model linear modules with
bitsandbytes LLM.int8. Vision, LoRA, multimodal, proprio, and continuous-action
components remain BF16.

Both result sets recorded:

```text
zero-equivalent arm actions: 0
full no-op actions: 0
repeated actions: 0
repeated chunks: 0
collapse warnings: 0
```

Detailed results are stored in each simulation project's `output/metrics/`
and `output/reports/` directories.

## Git branch rule

Task branches use:

```text
DF-number-number
```

Examples:

```text
DF-05-02
DF-06-01
DF-06-02
```

Only the active DF project should be staged and committed on its branch.


## Reproduction

Every experiment directory contains its own README and exact commands. Use the
environment, checkpoint, configuration, and launcher recorded inside that
directory rather than applying one global command to all experiments.
