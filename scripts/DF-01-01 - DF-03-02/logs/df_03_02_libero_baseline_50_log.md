# DF-03-02 LIBERO Baseline 50-Sample Run

## Run Scope

This 50-sample run is only used for DF-03-02 baseline pipeline and field-recording validation.
It checks fixed step selection, image/instruction extraction, OpenVLA inference, action parsing, latency logging, and VRAM logging.

Action values use `bridge_orig` because LIBERO-specific norm_stats were not found in the original OpenVLA config during DF-03-02 inspection.
Therefore action values are not official LIBERO evaluation results unless `unnorm_key_status` is confirmed.

Formal method evaluation will use the full test split / full-step open-loop evaluation later.
LIBERO simulation success rate is a separate rollout pipeline and is not measured in this run.

## Configuration

- Timestamp: `2026-06-11 08:19:05`
- Git branch: `DF-03-02`
- Git commit: `16fd738`
- Manifest JSON: `results/splits/df_03_02_baseline_50_samples_seed42.json`
- Dataset root: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds`
- Model ID: `openvla/openvla-7b`
- Unnorm key: `bridge_orig`
- Unnorm key status: `unconfirmed_for_libero`
- Action values official: `False`
- Pipeline debug only: `True`
- Device: `cuda:0`
- Dtype: `bfloat16`
- Warm-up runs: `1`
- Measured runs: `5`
- Fixed step policy: `middle`

## Runtime Environment

- Python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- Platform: `Linux-5.15.0-78-generic-x86_64-with-glibc2.35`
- Torch version: `2.11.0+cu128`
- Transformers version: `4.40.1`
- TensorFlow Datasets version: `4.9.4`
- CUDA available: `True`
- CUDA version: `12.8`
- GPU name: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- GPU total VRAM GB: `94.9708`

## Model Loading

- Processor load time seconds: `72.7439`
- Model load time seconds: `45.3224`
- VRAM before loading GB: `0.0000`
- VRAM after loading GB: `14.0905`

## Result Summary

- Total samples: `50`
- Action parse success count: `50`
- Action parse success rate: `1.0000`
- Action range valid count: `50`
- Action range valid rate: `1.0000`
- Inference error count: `0`
- Inference OOM count: `0`
- Overall average latency from sample averages: `128.2873 ms`
- Overall median latency from sample averages: `127.4796 ms`
- Peak VRAM allocated: `14.4252 GB`
- Peak VRAM reserved: `14.5059 GB`

## Output Files

- CSV metrics: `results/tables/df_03_02_libero_baseline_50_metrics.csv`
- Markdown log: `logs/df_03_02_libero_baseline_50_log.md`

## Important Interpretation

This run is pipeline and field-recording validation only.
It must not be used as the final method-comparison result.
Formal evaluation will be run later on the full test split / full-step open-loop evaluation.

