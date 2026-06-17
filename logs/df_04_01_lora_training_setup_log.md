# DF-04-01 LoRA Training Setup Log (batch=16)

## Purpose

This log records the setup and final result of DF-04-01 LoRA fine-tuning on the modified LIBERO RLDS dataset.

The goal of this task was to train an OpenVLA-7B LoRA adapter using the fixed LIBERO episode split created in DF-03-02.

## Final Status

DF-04-01 LoRA training finished successfully.

* Status: training_finished
* Return code: 0
* Total training steps: 200000
* Training time: 46.6366 hours
* Final train loss: 0.309302
* Final action accuracy: 0.901786
* Final L1 loss: 0.005042
* Final gradient norm: 3.7245857237893585
* Peak VRAM during logged training: about 56.97 GB

## Training Setup

* Model: OpenVLA-7B
* Method: LoRA fine-tuning
* Dataset: modified_libero_all_no_noops
* Train split: 1353 episodes
* Validation split: 167 episodes
* Test split: 173 episodes
* Seed: 42
* LoRA rank: 32
* LoRA alpha: 16
* LoRA dropout: 0.0
* Target modules: all-linear
* Batch size: 16
* Max steps: 200000
* Save steps: 5000
* Learning rate: 0.0005
* Gradient accumulation steps: 1
* Image augmentation: True
* Torch dtype: bfloat16
* GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition

## Files Modified or Added

### Project files

* `scripts/train_openvla_lora.py`
* `configs/df_04_01_lora_train_config.yaml`
* `logs/df_04_01_lora_training_setup_log.md`

### Modified official OpenVLA source files

* `/root/autodl-tmp/openvla/vla-scripts/finetune.py`
* `/root/autodl-tmp/openvla/prismatic/vla/datasets/rlds/dataset.py`
* `/root/autodl-tmp/openvla/prismatic/vla/datasets/rlds/oxe/mixtures.py`

These modified official files were downloaded to:

* `debug_lora_source/finetune.py`
* `debug_lora_source/rlds_dataset.py`
* `debug_lora_source/oxe_mixtures.py`

## Main Changes

### 1. Added combined LIBERO mixture

A new mixture named `modified_libero_all_no_noops` was added to `oxe_mixtures.py`.

It combines:

* `libero_spatial_no_noops`
* `libero_object_no_noops`
* `libero_goal_no_noops`
* `libero_10_no_noops`

Each subset uses sampling weight `1.0`.

### 2. Added fixed episode split support

`rlds/dataset.py` was modified to support:

```text
OPENVLA_EPISODE_SPLIT_JSON
```

This allows the official OpenVLA RLDS loader to use the fixed 80/10/10 episode-level split with seed 42.

The split file used was:

```text
results/splits/df_03_02_libero_episode_split_seed42.json
```

### 3. Switched official training launch to torchrun

The official OpenVLA `finetune.py` expects distributed training initialization.

Running it with plain Python caused this error:

```text
Default process group has not been initialized
```

The wrapper was changed to launch the official trainer through:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1
```

This fixed the DDP initialization issue.

### 4. Used local OpenVLA model cache

The cloud server could not access HuggingFace during training.

The model path was changed from:

```text
openvla/openvla-7b
```

to the local cached snapshot path:

```text
/root/autodl-tmp/hf_cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

### 5. Disabled W&B logging

The official script used W&B logging, but the cloud environment did not have a W&B API key.

W&B was disabled for this dissertation experiment. Local logs were used instead:

* Markdown log
* CSV summary
* Metadata JSON
* Full stdout log
* 500-step training records

### 6. Added 500-step console logging

`finetune.py` was modified to print training information every 500 steps.

Logged fields:

* step
* train_loss
* action_accuracy
* l1_loss
* learning_rate
* gradient_norm
* elapsed_time_hours
* peak_vram_gb

The full stdout log is saved as:

```text
experiments/02_lora/Pro_6000(Batch=16)/df_04_01_lora_training_log.stdout.log
```

The same 500-step records are also parsed into:

```text
experiments/02_lora/Pro_6000(Batch=16)/df_04_01_lora_training_metadata.json
```

## Problems Encountered and Fixes

### Problem 1: Wrong Python environment

The shell showed `(openvla310)`, but `which python` pointed to Python 3.12.

Fix:

```bash
export PATH=/root/miniconda3/envs/openvla310/bin:$PATH
hash -r
```

After fixing, the correct Python was used:

```text
/root/miniconda3/envs/openvla310/bin/python
Python 3.10.20
```

### Problem 2: Missing dependencies

Some required packages were missing, including:

* `peft`
* `draccus`

They were installed into the correct `openvla310` environment.

### Problem 3: DDP initialization error

Running the official `finetune.py` with plain Python caused DDP failure.

Fix:

The wrapper now launches the official trainer using `torchrun`.

### Problem 4: W&B login error

The official trainer required W&B login.

Fix:

W&B initialization and logging were disabled. Local logs were used instead.

## Training Result Summary

The training loss decreased clearly over the run.

Selected 500-step records:

|   Step | Train Loss | Action Accuracy |  L1 Loss |
| -----: | ---------: | --------------: | -------: |
|    500 |   3.134253 |        0.339286 | 0.194118 |
|  50000 |   1.657054 |        0.526786 | 0.045728 |
| 100000 |   0.776758 |        0.776786 | 0.012185 |
| 150000 |   0.440893 |        0.839286 | 0.008123 |
| 200000 |   0.309302 |        0.901786 | 0.005042 |

The final 20 logged points had an average train loss of about 0.3015 and average action accuracy of about 0.8857.

This shows that the LoRA adapter learned the training distribution. However, this does not directly prove task success in simulation. The real task success rate still needs to be evaluated later through LIBERO rollout.

## Final Artifacts

The final local artifact directory is:

```text
experiments/02_lora/Pro_6000(Batch=16)
```

Important files:

```text
df_04_01_lora_training_log.md
df_04_01_lora_training_log.stdout.log
df_04_01_lora_training_metadata.json
df_04_01_lora_training_summary.csv
resolved_df_04_01_lora_train_config.yaml
```

LoRA adapter:

```text
experiments/02_lora/Pro_6000(Batch=16)/lora_adapter
```

Contents:

```text
adapter_config.json
adapter_model.safetensors
README.md
```

Merged model:

```text
experiments/02_lora/Pro_6000(Batch=16)/merged_model
```

Important contents:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
config.json
tokenizer.json
tokenizer.model
processor_config.json
preprocessor_config.json
dataset_statistics.json
```

## Critical Notes and Limitations

This run only proves that LoRA training finished successfully and that the training loss decreased. It does not yet prove that the robot policy performs better in LIBERO simulation.

There is no validation loss in this run because the official training loop did not include a validation loop. Therefore, `best_val_loss` and `best_checkpoint_step` are recorded as null.

The model may still overfit the training data because it was trained for 200000 steps on 1353 training episodes. The final training loss is low, but task success must be checked separately.

The logged `action_accuracy` is a supervised training metric. It is not the same as task success rate. The real success rate must be measured in the later rollout evaluation stage.


