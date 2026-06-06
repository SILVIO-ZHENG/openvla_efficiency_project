# Remote GPU / HPC Setup Plan

## Purpose

This document describes how the remote GPU or university HPC environment is used in this OpenVLA efficiency project.

The project is designed around a local-development and remote-execution workflow. The local machine is mainly used for code editing, Git management, documentation, and lightweight debugging. The main OpenVLA model execution is planned to run on cloud GPU servers or university HPC because OpenVLA is a large Vision-Language-Action model and requires stronger GPU resources for reliable inference and fine-tuning.

## Project Context

This project investigates how to improve the efficiency of OpenVLA for robotic manipulation tasks.

The main efficiency methods include:

* LoRA-based fine-tuning
* QLoRA-based memory-efficient fine-tuning
* Quantized inference
* Latency measurement
* VRAM measurement
* Lightweight world-state / tiny world model extension

The remote GPU / HPC environment is used for the main model-related experiments, including OpenVLA baseline inference, LoRA / QLoRA fine-tuning, quantized inference testing, and experiment logging.

## Python Version Rule

This project uses:

```text
Python 3.10
```

All dependency installation and environment setup should be based on Python 3.10 compatibility.

Before installing dependencies on a remote GPU or HPC machine, the Python version should be checked first:

```bash
python --version
```

The expected result is:

```text
Python 3.10.x
```

If the remote machine does not provide Python 3.10 by default, a separate Python 3.10 environment should be created before installing the project dependencies.

## Local Machine Role

The local machine is mainly used for:

* Code editing
* Git management
* Project documentation
* Lightweight debugging
* Small utility scripts

The local laptop is not expected to run the full OpenVLA model for the main experiments.

## Remote GPU / HPC Role

The remote GPU or university HPC system is used for:

* OpenVLA baseline inference
* LoRA fine-tuning
* QLoRA fine-tuning
* Quantized inference testing
* Latency measurement
* VRAM measurement
* Experiment logging
* Model evaluation

This separation makes the workflow more practical. The local computer keeps the project organised, while the remote GPU / HPC provides the compute power needed for OpenVLA.

## Remote Execution Workflow

The planned remote execution workflow is:

```text
Local development
        ↓
Push code to GitHub
        ↓
Pull code on remote GPU / HPC
        ↓
Create Python 3.10 environment
        ↓
Install dependencies
        ↓
Run OpenVLA baseline
        ↓
Run LoRA / QLoRA / quantization experiments
        ↓
Save logs and results
        ↓
Pull small results back to local machine
```

## Repository Usage on Remote GPU / HPC

The GitHub repository is used to manage:

* Source code
* Configuration files
* Documentation
* Small result tables
* Small figures

The remote machine will clone the repository from GitHub:

```bash
git clone https://github.com/SILVIO-ZHENG/openvla_efficiency_project.git
cd openvla_efficiency_project
```

Large files should not be committed to GitHub.

## Recommended Remote Storage Structure

A suggested remote storage structure is:

```text
~/openvla_workspace/
├── repo/
│   └── openvla_efficiency_project/
├── datasets/
├── models/
├── checkpoints/
└── logs/
```

The repository folder stores the project code. Large files such as datasets, model weights, checkpoints, and logs should be stored outside the Git repository.

## Files Not Stored in GitHub

The following files should not be committed to GitHub:

* OpenVLA model weights
* Checkpoints
* Raw datasets
* Large processed datasets
* Large experiment logs
* Temporary training outputs
* GPU / HPC job output files
* Private environment files such as `.env`

These files should stay on remote GPU / HPC storage or external storage.

## Planned Remote Environment Components

The remote GPU / HPC environment is expected to support:

* Python 3.10
* CUDA-enabled PyTorch
* Transformers
* Accelerate
* PEFT
* BitsAndBytes
* OpenVLA dependencies
* Experiment logging tools

Heavy OpenVLA-related dependencies should be installed on the remote GPU / HPC machine rather than on the local laptop.

## Experiment Stages on Remote GPU / HPC

### Stage 1: Environment Check

The first remote stage checks whether the GPU and Python environment are ready.

This includes checking:

* Python version
* GPU availability
* CUDA availability
* PyTorch GPU support

### Stage 2: OpenVLA Baseline

The second stage runs the original OpenVLA model without LoRA, QLoRA, or quantization.

The baseline records:

* Model loading time
* Inference latency
* VRAM usage
* Output action format
* Basic task performance

This baseline is important because all later efficiency methods will be compared against it.

### Stage 3: LoRA Fine-Tuning

The LoRA stage fine-tunes OpenVLA by training a small number of adapter parameters while keeping most of the base model frozen.

This stage evaluates whether LoRA can reduce fine-tuning cost while maintaining task performance.

### Stage 4: QLoRA Fine-Tuning

The QLoRA stage reduces GPU memory usage further by loading the base model in a low-bit format and training LoRA adapters on top.

This stage is used to test whether OpenVLA can be adapted under more limited GPU resources.

### Stage 5: Quantized Inference

The quantization stage compares lower-precision inference settings such as FP16, INT8, and INT4.

The main evaluation metrics are:

* Inference latency
* VRAM usage
* Model size
* Performance change after quantization

### Stage 6: Lightweight World-State / Tiny World Model Extension

The lightweight world-state or tiny world model module is an optional extension.

It does not replace OpenVLA. Instead, it is used as a small auxiliary module to represent the current scene or check whether a candidate action is likely to help the task.

This module is included to test whether a small predictive component can improve action stability without adding too much computational cost.

## Summary

This project uses a local-development and remote-execution structure.

The local machine is used to organise the project, write code, manage Git, and prepare documentation. The remote GPU / HPC environment is used to run the main OpenVLA experiments.

This setup supports the main goal of the dissertation: evaluating how LoRA, QLoRA, quantization, and a lightweight world-state extension can improve the efficiency and deployment feasibility of OpenVLA for robotic manipulation.
