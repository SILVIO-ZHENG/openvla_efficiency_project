# DF-06-02 LoRA+OFT 4-bit NF4 LIBERO-10 Simulation

I use this project to evaluate the DF-04-03 LoRA+OFT best checkpoint with an
NF4-quantized language model in closed-loop LIBERO-10 simulation.

Only eligible language-model linear layers are quantized. The vision
backbone, multimodal projector, LoRA adapter, `lm_head`, proprio projector,
and continuous L1 action head remain in BF16.

## Model contract

- Training run:
  `df_04_03_lora_oft_libero10_seed42_rank32_bs12_100k`
- Checkpoint: best adapter at step 100,000
- Training objective: continuous L1 regression
- Policy output: `[8, 7]` continuous action chunk
- Inputs: primary image, wrist image, and normalized 8-D proprio
- Dataset: `libero_10_no_noops`
- LoRA: rank 32, alpha 16, dropout 0
- Quantization: bitsandbytes NF4 with double quantization
- NF4 compute dtype: BF16
- Quantized linear modules: 224
- CPU and disk offload: disabled
- Discrete action-token fallback: disabled

NF4 is a 4-bit floating quantization codebook. This is mixed-precision
post-training inference quantization, not QLoRA training or full-model INT4.

## Runtime environment

text
GPU: NVIDIA GeForce RTX 5090, 32 GB
Python: 3.10.20
PyTorch: 2.11.0+cu128
CUDA: 12.8
Transformers: 4.40.1
PEFT: 0.11.1
bitsandbytes: 0.49.2
Accelerate: 0.29.3
LIBERO: 0.1.0
Robosuite: 1.4.0
MuJoCo: 3.3.2


## Evaluation protocol

- Suite: LIBERO-10
- Tasks: 0–9
- Trials per task: 50
- Total rollouts: 500
- Environment seed: 0
- Experiment seed: 42
- Initial settling steps: 10
- Maximum policy-controlled steps: 520
- One inference returns eight actions
- Success checked after every individual action
- Stop immediately after success
- Save successful and failed rollout videos

The evaluation uses the same preprocessing, checkpoint, normalization
statistics, task states, seeds, and action execution rules as DF-05-02 BF16.

## Formal results

| Metric | Result |
|---|---:|
| Total rollouts | 500 |
| Successful rollouts | 466 |
| Failed rollouts | 34 |
| Success rate | 93.2% |
| Timeout count | 34 |
| Average successful steps | 250.751 |
| Average inference latency | 154.774 ms |
| P95 inference latency | 157.683 ms |
| Amortized inference latency per action | 19.602 ms |
| Average end-to-end latency | 299.643 ms |
| Average control frequency | 26.351 Hz |
| Peak allocated VRAM | 6.288 GiB |
| Peak reserved VRAM | 6.953 GiB |
| Model memory footprint | 5.237 GiB |
| Model load time | 8.591 seconds |
| Total runtime | 1.685 hours |

## Per-task results

| Task | Success | Failure | Rate | Avg. success steps |
|---:|---:|---:|---:|---:|
| 0 | 50 | 0 | 100% | 276.64 |
| 1 | 49 | 1 | 98% | 253.51 |
| 2 | 49 | 1 | 98% | 244.16 |
| 3 | 49 | 1 | 98% | 226.76 |
| 4 | 44 | 6 | 88% | 226.34 |
| 5 | 48 | 2 | 96% | 188.75 |
| 6 | 45 | 5 | 90% | 221.64 |
| 7 | 48 | 2 | 96% | 241.29 |
| 8 | 36 | 14 | 72% | 396.50 |
| 9 | 48 | 2 | 96% | 264.00 |

All 34 failed rollouts terminated by reaching the 520-step timeout.

## BF16 versus NF4

| Metric | BF16 | NF4 | Change |
|---|---:|---:|---:|
| Success rate | 94.0% | 93.2% | -0.8 percentage points |
| Average inference latency | 135.889 ms | 154.774 ms | +13.8978% |
| P95 inference latency | 148.845 ms | 157.683 ms | +5.9378% |
| Average end-to-end latency | 294.480 ms | 299.643 ms | +1.7532% |
| Control frequency | 26.808 Hz | 26.351 Hz | -1.7034% |
| Peak allocated VRAM | 15.193 GiB | 6.288 GiB | -58.6137% |
| Peak reserved VRAM | 15.508 GiB | 6.953 GiB | -55.1637% |

Complete comparison files:

text
output/reports/bf16_vs_nf4_comparison.csv
output/reports/bf16_vs_nf4_comparison.json
output/reports/bf16_vs_nf4_comparison.md


## Collapse diagnostics

text
zero-equivalent arm actions: 0
full no-op actions: 0
repeated actions: 0
repeated chunks: 0
collapse warnings: 0
discrete action-token decoding used: false
simulation-side action intervention enabled: false


The inference path was:

text
continuous_l1_action_head_nf4_language_model


## Output structure

text
output/
├── logs/
├── metrics/
│   ├── failure_cases.csv
│   ├── inference_events.jsonl
│   ├── resolved_config.yaml
│   ├── rollout_results.csv
│   ├── runtime_manifest.json
│   ├── simulation_summary.json
│   └── task_summary.csv
├── reports/
│   ├── int4_model_audit.json
│   ├── bf16_vs_nf4_comparison.*
│   ├── remote_file_inventory.tsv
│   └── remote_SHA256SUMS.txt
└── videos/
    └── task_XX_trial_YYY.mp4


`rollout_results.csv` records every rollout's task, trial, success status,
`episode_steps`, and `steps_to_success`.

The formal run produced 500 non-empty videos with a total size of
approximately 0.671 GiB. MP4 files are retained locally and excluded from
Git.

## Validation

text
NF4 CUDA kernel validation: PASS
NF4 model audit: PASS
Quantized Linear4bit modules: 224
50-step smoke test: PASS
500-rollout simulation: PASS
Formal videos: 500
Empty videos: 0
Local output validation: PASS


The output archive was:

text
df0602_int4_simulation_output_20260729T032656Z.tar.gz
SHA256: 9119991b764535b47e9f61339001a48c919cfc9dee04a2182e5341441a0fde77


## Reproduction

Local validation:

powershell
powershell -ExecutionPolicy Bypass -File `
".\scripts\Simulation\DF-06-02 LoRA+OFT INT4\launch\run_simulation.ps1"


Cloud model audit:

bash
/root/miniconda3/envs/openvla310/bin/python \
diagnostics/int4_model_audit.py \
--config config/simulation_config.yaml


Formal simulation:

bash
bash launch/run_simulation.sh


## Status

text
DF-06-02 completed
500 rollouts recorded
Formal output downloaded and validated
