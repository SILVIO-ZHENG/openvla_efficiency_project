# DF-06-01: LoRA + OFT INT8 — LIBERO-10 Simulation

This project evaluates the **DF-04-03 LoRA + OFT best checkpoint** in
closed-loop LIBERO-10 simulation.

> **Quantization scope:** Eligible language-model linear layers use
> bitsandbytes INT8. Vision, LoRA, multimodal, proprio, and continuous-action
> components remain BF16.

## Experiment configuration

| Field | Value |
| :--- | :--- |
| Training run | `df_04_03_lora_oft_libero10_seed42_rank32_bs12_100k` |
| Checkpoint | Best adapter, step 100,000 |
| Action head | Continuous L1 regression |
| Action output | `[8, 7]` per inference |
| Inputs | Primary image, wrist image, normalized 8-D proprio |
| Dataset | `libero_10_no_noops` |
| Split | Train 330, validation 10, test 39, seed 42 |
| Quantization | bitsandbytes `llm_int8` |
| Quantized component | Eligible language-model linear layers |
| Non-quantized dtype | BF16 |
| LoRA | Rank 32, alpha 16, dropout 0 |
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| Python | 3.10.20 |
| PyTorch / CUDA | 2.11.0+cu128 / 12.8 |
| Transformers / PEFT | 4.40.1 / 0.11.1 |
| bitsandbytes / Accelerate | 0.49.2 / 0.29.3 |
| LIBERO / Robosuite / MuJoCo | 0.1.0 / 1.4.0 / 3.3.2 |

## Evaluation protocol

- **Tasks:** LIBERO-10 tasks 0–9
- **Trials:** 50 per task; 500 rollouts in total
- **Initial states:** Official deterministic initial states
- **Seeds:** Environment seed 0; experiment seed 42
- **Initial settling:** 10 steps
- **Maximum episode length:** 520 policy-controlled steps
- **Action chunking:** One model query per eight actions
- **Success checking:** After every environment action
- **Early stopping:** Stop immediately on success
- **Video recording:** Successful and failed videos saved

## Formal INT8 results

### Outcome summary

| Metric | Value |
| :--- | ---: |
| Total rollouts | 500 |
| Successful rollouts | 474 |
| Failed rollouts | 26 |
| **Overall success rate** | **94.8%** |
| Timeout count | 26 |
| Average successful-rollout steps | 252.084 |
| Average steps across all rollouts | 266.016 |
| Completed | `true` |
| Process exit code | 0 |

### Runtime and latency

| Metric | Value |
| :--- | ---: |
| Average inference latency | 199.351 ms |
| P95 inference latency | 209.467 ms |
| Amortized inference latency | 25.265 ms/action |
| Average end-to-end latency | 345.041 ms |
| Amortized end-to-end latency | 43.729 ms/action |
| Average control frequency | 22.868 Hz |
| Model load time | 8.942 s |
| Rollout runtime | 1.882 h |

### Memory usage

| Metric | Value |
| :--- | ---: |
| Peak allocated VRAM | 9.263 GiB |
| Peak reserved VRAM | 9.787 GiB |
| Model memory footprint | 8.253 GiB |

### Per-task results

| Task | Successes | Failures | Success rate |
| ---: | ---: | ---: | ---: |
| 0 | 50 | 0 | 100% |
| 1 | 47 | 3 | 94% |
| 2 | 50 | 0 | 100% |
| 3 | 49 | 1 | 98% |
| 4 | 45 | 5 | 90% |
| 5 | 46 | 4 | 92% |
| 6 | 46 | 4 | 92% |
| 7 | 50 | 0 | 100% |
| 8 | 43 | 7 | 86% |
| 9 | 48 | 2 | 96% |

## Runtime audit and diagnostics

| Field | Value |
| :--- | :--- |
| INT8 model flag | `true` |
| Quantization audit passed | `true` |
| Quantized `Linear8bitLt` modules | 224 |
| Continuous inference path | `continuous_l1_action_head_int8_language_model` |
| Discrete action decoding used | `false` |
| Zero-equivalent arm actions | 0 |
| Full no-op actions | 0 |
| Repeated actions | 0 |
| Repeated chunks | 0 |
| Collapse warnings | 0 |
| Action intervention enabled | `false` |

## Recorded BF16 and INT8 comparison

| Metric | BF16 | INT8 | INT8 − BF16 |
| :--- | ---: | ---: | ---: |
| Success rate | 94.0% | 94.8% | +0.8 percentage points |
| Timeout rate | 6.0% | 5.2% | −0.8 percentage points |
| Average inference latency | 135.889 ms | 199.351 ms | +63.462 ms |
| P95 inference latency | 148.845 ms | 209.467 ms | +60.622 ms |
| Amortized inference latency | 17.214 ms/action | 25.265 ms/action | +8.052 ms/action |
| Amortized end-to-end latency | 37.303 ms/action | 43.729 ms/action | +6.427 ms/action |
| Average control frequency | 26.808 Hz | 22.868 Hz | −3.940 Hz |
| Peak allocated VRAM | 15.193 GiB | 9.263 GiB | −5.930 GiB |
| Peak reserved VRAM | 15.508 GiB | 9.787 GiB | −5.721 GiB |

## Recorded artifacts

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
│   ├── bf16_vs_int8_comparison.csv
│   ├── bf16_vs_int8_comparison.json
│   ├── bf16_vs_int8_comparison.md
│   ├── formal_run_validation.json
│   └── int8_model_audit.json
└── videos/
    └── task_XX_trial_YYY.mp4
```

| Video artifact | Value |
| :--- | ---: |
| Formal videos | 500 |
| Empty videos | 0 |
| Total formal video size | 0.666 GiB |

## Run

```bash
cd "/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-06-01 LoRA+OFT INT8"
bash launch/run_simulation.sh
```
