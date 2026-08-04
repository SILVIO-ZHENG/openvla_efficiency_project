# OpenVLA Efficiency Project

This repository contains an end-to-end OpenVLA fine-tuning, deployment,
diagnostic, and closed-loop evaluation pipeline for robotic manipulation.

The project evaluates parameter-efficient training and mixed-precision
deployment on LIBERO-10, with primary-camera images, wrist-camera images,
language instructions, and robot proprioception.

## Project status

```text
Core experiments completed
```

## Project scope

- OpenVLA LoRA fine-tuning
- OpenVLA LoRA+OFT continuous-action fine-tuning
- RLDS dataset preprocessing and deterministic dataset splitting
- Primary-camera, wrist-camera, and proprioception integration
- Continuous `[8, 7]` action-chunk prediction
- Closed-loop LIBERO-10 simulation
- BF16 inference evaluation
- Mixed-precision INT8 inference evaluation
- Mixed-precision 4-bit NF4 inference evaluation
- Latency, control-frequency, VRAM, and action-collapse recording
- Offline action-distribution and failure diagnostics
- Cross-precision deployment comparison
- QLoRA evaluation
- Lightweight world-state extension

## System pipeline

```text
Language instruction
        +
Primary camera image
        +
Wrist camera image
        +
8-D robot proprioception
        ↓
OpenVLA multimodal backbone
        ↓
LoRA-adapted language model
        ↓
OFT continuous action head
        ↓
Eight 7-D robot actions
        ↓
LIBERO closed-loop execution
```

Each predicted action contains:

```text
[x, y, z, roll, pitch, yaw, gripper]
```

The policy returns an action chunk with shape:

```text
[8, 7]
```

## Dataset and training setup

The dataset pipeline audited four modified LIBERO RLDS subsets containing
1,693 trajectories.

The formal LoRA+OFT experiment uses `libero_10_no_noops` with the following
fixed split:

| Split | Episodes |
|---|---:|
| Training | 330 |
| Validation | 10 |
| Test | 39 |
| Total | 379 |

Main training configuration:

| Setting | Value |
|---|---|
| Base model | `openvla/openvla-7b` |
| Training objective | Continuous L1 regression |
| Action chunk | `[8, 7]` |
| LoRA rank | 32 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Batch size | 12 |
| Training steps | 100,000 |
| Best checkpoint | Step 100,000 |
| Best validation loss | 0.106282 |
| Training precision | BF16 |
| Dataset split seed | 42 |

The trained checkpoint contains:

```text
adapter_model.safetensors
adapter_config.json
action_head.pt
proprio_projector.pt
dataset_statistics.json
processor/
```

## Repository structure

```text
scripts/
├── Training/
│   ├── DF-04-02 LoRA/
│   │   └── Discrete-action LoRA training
│   └── DF-04-03 Lora+OFT/
│       └── Continuous-action LoRA+OFT training
│
└── Simulation/
    ├── DF-05-01 Lora_Batch12/
    │   └── Discrete LoRA evaluation and offline diagnostics
    ├── DF-05-02 Lora+OFT/
    │   └── LoRA+OFT BF16 evaluation
    ├── DF-06-01 LoRA+OFT INT8/
    │   └── Mixed-precision LLM.int8 evaluation
    └── DF-06-02 LoRA+OFT INT4/
        └── Mixed-precision 4-bit NF4 evaluation

docs/                       Project documentation
data/                       Dataset placeholders and metadata
results/                    Cross-experiment tables and figures
```

Each DF project contains its own configuration, source code, launch scripts,
README, logs, metrics, diagnostics, and reports.

## Experiment map

| ID | Experiment | Status |
|---|---|---|
| DF-04-02 | Discrete-action OpenVLA LoRA training | Completed |
| DF-04-03 | Continuous-action OpenVLA LoRA+OFT training | Completed |
| DF-05-01 | LoRA closed-loop evaluation and diagnostics | Completed |
| DF-05-02 | LoRA+OFT BF16 closed-loop evaluation | Completed |
| DF-06-01 | LoRA+OFT mixed-precision INT8 evaluation | Completed |
| DF-06-02 | LoRA+OFT mixed-precision NF4 evaluation | Completed |

## Closed-loop evaluation protocol

All main LoRA+OFT evaluations use the same protocol:

| Setting | Value |
|---|---|
| Benchmark | LIBERO-10 |
| Number of tasks | 10 |
| Trials per task | 50 |
| Total rollouts | 500 |
| Maximum controlled steps | 520 |
| Initial settling steps | 10 |
| Environment seed | 0 |
| Initial states | Official deterministic states |
| Model query frequency | Once per eight-action chunk |
| Success check | After every environment action |
| Early termination | Immediately after success |

## Recorded closed-loop results

All three experiments use the DF-04-03 step-100,000 adapter and the same
500-rollout protocol.

| Experiment | Precision | Success | Avg. query latency | Amortized inference/action | Control rate | Peak allocated VRAM |
|---|---|---:|---:|---:|---:|---:|
| DF-05-02 | BF16 | 470/500 (94.0%) | 135.889 ms | 17.214 ms | 26.808 Hz | 15.193 GiB |
| DF-06-01 | INT8 + BF16 | 474/500 (94.8%) | 199.351 ms | 25.265 ms | 22.868 Hz | 9.263 GiB |
| DF-06-02 | NF4 + BF16 | 466/500 (93.2%) | 154.774 ms | 19.602 ms | 26.351 Hz | 6.288 GiB |

`Avg. query latency` measures one model call that produces an eight-action
chunk. `Amortized inference/action` divides inference cost across the actions
that are actually executed.

## Quantization scope

DF-06-01 and DF-06-02 use mixed-precision inference rather than quantizing
every model component.

| Component | BF16 | INT8 | NF4 |
|---|---|---|---|
| Eligible language-model linear layers | BF16 | LLM.int8 | 4-bit NF4 |
| Quantized linear modules | 0 | 224 | 224 |
| Vision backbone | BF16 | BF16 | BF16 |
| LoRA adapter | BF16 | BF16 | BF16 |
| Multimodal projector | BF16 | BF16 | BF16 |
| Proprio projector | BF16 | BF16 | BF16 |
| Continuous action head | BF16 | BF16 | BF16 |

The INT8 and NF4 experiments use the same trained LoRA+OFT adapter. Quantization
is applied to eligible base language-model linear layers during model loading.

## Action-collapse diagnostics

DF-05-01 identified discrete zero-action token collapse in the original LoRA
baseline. Offline diagnostics showed weak complete-action accuracy and an
increase in no-op predictions on the validation distribution.

The LoRA+OFT continuous-action evaluations therefore record:

```text
zero-equivalent arm actions
full no-op actions
repeated actions
repeated chunks
maximum consecutive zero-action streak
maximum repeated-chunk streak
collapse warnings
```

The BF16, INT8, and NF4 formal evaluations all recorded:

```text
zero-equivalent arm actions: 0
full no-op actions: 0
repeated actions: 0
repeated chunks: 0
collapse warnings: 0
```

These checks are diagnostic only. They do not modify, replace, or suppress
predicted actions.

## Recorded outputs

Each simulation experiment can produce:

```text
output/
├── logs/
│   ├── full_console.log
│   └── simulation.log
├── metrics/
│   ├── failure_cases.csv
│   ├── inference_events.jsonl
│   ├── resolved_config.yaml
│   ├── rollout_results.csv
│   ├── runtime_manifest.json
│   ├── simulation_summary.json
│   └── task_summary.csv
├── reports/
│   ├── model audit reports
│   ├── formal-run validation reports
│   └── cross-precision comparison reports
└── videos/
    └── task_XX_trial_YYY.mp4
```

The recorded fields include:

- Per-rollout success and termination reason
- Steps required for successful rollouts
- Per-task success rate
- Model-query and end-to-end latency
- Amortized latency per executed action
- Control frequency
- Peak allocated and reserved VRAM
- Model memory footprint
- Predicted and executed action chunks
- Quantization backend and module audit
- Action-collapse diagnostics
- Runtime environment and resolved configuration

## Environment

All project code targets:

```text
Python 3.10
```

Main verified simulation environment:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 4.40.1 |
| PEFT | 0.11.1 |
| bitsandbytes | 0.49.2 |
| Accelerate | 0.29.3 |
| LIBERO | 0.1.0 |
| Robosuite | 1.4.0 |
| MuJoCo | 3.3.2 |

Main workloads:

| Workload | Environment |
|---|---|
| Code editing and Git | Local Windows workstation |
| Model training | Cloud GPU / university HPC |
| Closed-loop simulation | NVIDIA RTX 5090 |
| Simulator | LIBERO-10 |

## Git branch rule

Task branches use:

```text
DF-number-number
```

Examples:

```text
DF-04-03
DF-05-02
DF-06-01
DF-06-02
```

Only the active DF project should be staged and committed on its corresponding
branch.

## Artifact policy

The Git repository does not store:

- OpenVLA base-model weights
- Raw RLDS datasets
- Large adapter archives
- Formal rollout MP4 files
- Python cache files
- Individual files exceeding GitHub's hard file-size limit

Git retains the source code, configurations, structured metrics, summaries,
audit reports, and selected logs required for reproducibility.

Model artifacts are stored separately and can be distributed through the
Hugging Face Hub.

## Reproduction

Every experiment directory contains its own README and exact commands.

Reproduction should use the environment, checkpoint, dataset statistics,
configuration, and launcher stored in the corresponding DF directory rather
than applying one global command to every experiment.

The OpenVLA base model must be obtained separately from:

```text
openvla/openvla-7b
```

The trained LoRA adapter, continuous action head, proprio projector, processor,
and dataset statistics must all be available before running LoRA+OFT inference.