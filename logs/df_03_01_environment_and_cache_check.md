# DF-03-01 Environment and Cache Check

Note: The following paths are local paths on the author's cloud machine. They are recorded for experiment reproducibility and do not contain access credentials.

## Purpose

This file records the cloud environment, model cache, and dataset cache after DF-03-01 preparation.

The goal is to keep the environment stable before moving to DF-03-02 baseline evaluation.

## Important Rule

Do not randomly upgrade core packages such as:

* transformers
* tokenizers
* huggingface_hub
* torch
* datasets

The OpenVLA environment should stay fixed unless a later task clearly requires a change.

## Python Environment

Python executable: /root/miniconda3/envs/openvla310/bin/python
Python version: 3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]

## Key Package Versions

torch: 2.11.0+cu128
transformers: 4.40.1
tokenizers: 0.19.1
huggingface_hub: 0.23.4
datasets: 2.19.1
tensorflow: 2.15.1
tensorflow_datasets: 4.9.4

## CUDA / GPU

CUDA available: False
CUDA version: 12.8

## Cache Paths

/root/autodl-tmp/hf_cache/hub/models--openvla--openvla-7b: exists=True, is_dir=True
/root/autodl-tmp/datasets/openvla_modified_libero_rlds: exists=True, is_dir=True

## Model and Dataset Size

15G	/root/autodl-tmp/hf_cache/hub/models--openvla--openvla-7b
9.6G	/root/autodl-tmp/datasets/openvla_modified_libero_rlds

## Dataset Structure Check

The local modified LIBERO RLDS dataset was found under:

`/root/autodl-tmp/datasets/openvla_modified_libero_rlds`

The dataset contains four local TFDS-style sub-datasets:

| Sub-dataset             | Version path                                                                           | Episodes |
| ----------------------- | -------------------------------------------------------------------------------------- | -------: |
| libero_10_no_noops      | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_10_no_noops/1.0.0`      |      379 |
| libero_goal_no_noops    | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_goal_no_noops/1.0.0`    |      428 |
| libero_object_no_noops  | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_object_no_noops/1.0.0`  |      454 |
| libero_spatial_no_noops | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_spatial_no_noops/1.0.0` |      432 |

Total number of episodes:

`1693 episodes`

Only the `train` split was confirmed during this check.

## Step Count Planning Assumption

A small step-count sample was checked on `libero_goal_no_noops`.

Sample result:

| Item                                             |  Value |
| ------------------------------------------------ | -----: |
| sampled episodes                                 |     30 |
| sampled total steps                              |   3547 |
| average steps per sampled episode                | 118.23 |
| estimated total steps for `libero_goal_no_noops` |  50603 |

For conservative planning, this project assumes:

* 150–180 steps per episode
* approximately 250,000–300,000 total trajectory steps

This estimate is used only for planning. It should not be treated as the exact full step count.

## Status

DF-03-01 environment and cache preparation is complete.

The local OpenVLA model cache exists.

The local modified LIBERO RLDS dataset cache exists.

The dataset contains 1693 confirmed episodes.

The next task is DF-03-02 baseline evaluation.
