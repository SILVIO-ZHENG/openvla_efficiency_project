# DF-02-02 Baseline Inference Log

## Purpose

This log records the baseline inference logging plan for the original OpenVLA model before applying LoRA, QLoRA, quantization, or world-model extensions.

The purpose of DF-02-02 was to define what information should be recorded during baseline inference. It also prepared the structure for later cloud GPU testing.

At this stage, the local machine was only used for script checking, Git management, and dry-run validation. Full OpenVLA inference was planned to run on cloud GPU or university HPC because the local laptop GPU is not suitable for loading and running the full OpenVLA-7B model.

## Baseline Method

* Model: OpenVLA baseline
* Fine-tuning: None
* Quantization: None
* Extra world model: None
* Expected execution environment: Cloud GPU or university HPC
* Local machine role: script checking, dry-run validation, and Git management only

## What DF-02-02 Did

DF-02-02 did not produce the final baseline inference result.

Instead, it prepared the baseline inference recording structure, including:

* environment information to record
* model loading information to record
* inference latency fields
* VRAM usage fields
* sample action output fields
* action parsing fields
* local dry-run note

This task helped define the baseline logging format before running the real cloud smoke test in DF-02-03.

## Metrics to Record

### Environment

The following environment information should be recorded when full baseline inference is executed:

* Git commit hash
* Branch
* Python version
* PyTorch version
* Transformers version
* CUDA available
* CUDA version
* GPU name
* Total VRAM

### Model Loading

The following model loading information should be recorded:

* Model ID
* Dtype
* Device
* Model loading time
* VRAM before loading
* VRAM after loading

### Inference Latency

The following latency information should be recorded:

* Number of warm-up runs
* Number of measured runs
* Average inference latency
* Median inference latency
* Minimum inference latency
* Maximum inference latency
* Standard deviation

### VRAM Usage

The following GPU memory information should be recorded:

* VRAM before inference
* VRAM after inference
* Peak VRAM allocated
* Peak VRAM reserved

### Sample Action Output

The following sample output information should be recorded:

* Instruction
* Image path
* Raw model output
* Parsed action
* Action dimension
* Action parse success

## Local Dry Run Result

The local dry run was only used to check script logic and logging structure.

The local dry run did not load the full OpenVLA model and did not produce final baseline inference metrics.

The local machine was used only to verify that the project structure, script arguments, and expected output fields were reasonable before moving to cloud execution.

## Relationship with DF-02-03

DF-02-03 continued this work by running a cloud smoke test with the OpenVLA baseline model.

DF-02-03 produced actual cloud GPU outputs, including:

* sample action outputs
* action parse success status
* inference latency
* VRAM usage
* CSV result file
* Markdown log file

Therefore, DF-02-02 should be understood as the baseline logging preparation step, while DF-02-03 is the first successful cloud baseline smoke test.

## Notes

The baseline result will be compared with later LoRA, QLoRA, and quantized inference results.

This file should not report fake latency, fake VRAM, or fake action results. Only measured results from cloud GPU or HPC should be used for formal experiment reporting.
