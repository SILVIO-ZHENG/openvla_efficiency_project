# DF-03-01 Dataset and Baseline Evaluation Plan

Note: The following paths are local paths on the author's cloud machine. They are recorded for experiment reproducibility and do not contain access credentials.

## 1. Purpose

This document defines the dataset choice and baseline evaluation plan for the next stage of the OpenVLA efficiency project.

The previous task, DF-02-03, verified that the OpenVLA baseline model can run on the cloud GPU with 10 manually selected sample images. That task was a cloud smoke test. It proved that the model, environment, image input, action output, latency logging, VRAM logging, CSV output, and Markdown log output can work on the cloud machine.

The next step is to move from a simple smoke test to a more structured dataset-based evaluation setup.

The goal of DF-03-01 is not to train the model yet. Instead, this task defines the dataset path, evaluation input format, output metrics, sample count, split strategy, action parsing rule, dataset size, and the reason for starting with a smaller OpenVLA-related dataset before moving to larger real-world robot datasets.

## 2. Dataset Choice

The first dataset selected for the next stage is the OpenVLA modified LIBERO RLDS dataset.

Dataset name:

`openvla/modified_libero_rlds`

Cloud storage path:

`/root/autodl-tmp/datasets/openvla_modified_libero_rlds`

This dataset is selected because it is directly connected with OpenVLA fine-tuning experiments and uses the RLDS robot learning data format. It is also much smaller and easier to manage than the larger real-world robot dataset planned for later extension.

This dataset is suitable for the next project stage because the current goal is to build a stable and repeatable evaluation pipeline before scaling to larger data.

## 2.1 Confirmed Dataset Structure

The local modified LIBERO RLDS dataset has been checked on the cloud machine.

Dataset root:

`/root/autodl-tmp/datasets/openvla_modified_libero_rlds`

The local dataset contains four TFDS-style sub-datasets:

| Sub-dataset             | Local version path                                                                     | Episodes |
| ----------------------- | -------------------------------------------------------------------------------------- | -------: |
| libero_10_no_noops      | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_10_no_noops/1.0.0`      |      379 |
| libero_goal_no_noops    | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_goal_no_noops/1.0.0`    |      428 |
| libero_object_no_noops  | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_object_no_noops/1.0.0`  |      454 |
| libero_spatial_no_noops | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_spatial_no_noops/1.0.0` |      432 |

Total:

`1693 episodes`

All four checked sub-datasets currently expose a `train` split. No separate validation split was confirmed during this check. Therefore, the first baseline evaluation should use a fixed subset from the available `train` split.

## 2.2 Step Count Planning Assumption

A small step-count sample was checked on `libero_goal_no_noops`.

Sample result:

| Item                                             |  Value |
| ------------------------------------------------ | -----: |
| sampled episodes                                 |     30 |
| sampled total steps                              |   3547 |
| average steps per sampled episode                | 118.23 |
| estimated total steps for `libero_goal_no_noops` |  50603 |

For conservative planning, this project assumes 150–180 steps per episode.

Estimated total trajectory steps:

`1693 × 150–180 ≈ 250,000–300,000 steps`

This estimate is used for runtime and cost planning. It should not be treated as the exact full step count.

## 3. Why Not Start with the 124GB Dataset

The larger 124GB dataset is not used as the first formal dataset because it brings higher engineering risk at this stage.

The main risks are:

* large storage requirement
* slow download speed
* higher chance of broken or incomplete downloads
* longer preprocessing time
* more difficult data loader debugging
* higher cloud GPU cost if training starts too early
* more difficult experiment recovery if something fails

The current project focus is OpenVLA efficiency improvement, including baseline inference, LoRA, QLoRA, and quantized inference. Therefore, the first priority is to build a stable evaluation pipeline before scaling to a larger dataset.

The 124GB dataset will be considered later as an extension stage. The first use of that dataset should be a small verified subset, not the full dataset.

## 4. Project Stage Definition

The current project plan is divided as follows.

### Stage 1: Small Dataset Evaluation

This stage uses the modified LIBERO RLDS dataset to build a repeatable evaluation pipeline.

Main goals:

* verify dataset download
* inspect dataset structure
* read a small number of samples
* connect dataset samples to OpenVLA input format
* record latency, VRAM, and action output
* save results to CSV and Markdown logs

### Stage 2: Larger Dataset Subset

This stage may use a small subset of the larger 124GB real-world robot dataset.

Main goals:

* check whether the same evaluation pipeline can work on real-world robot data
* avoid the cost and risk of full-dataset training at the beginning
* provide stronger evidence that the method is not limited to a small benchmark

### Stage 3: Full Larger Dataset

This stage is optional.

It will only be considered if compute, storage, and time are enough. It is not required for the first complete version of the MSc dissertation experiment.

## 5. Baseline Evaluation Design

The baseline evaluation should use the same structure for future comparisons.

The same input format and metrics should be reused for:

* OpenVLA baseline
* LoRA fine-tuned OpenVLA
* QLoRA fine-tuned OpenVLA
* quantized OpenVLA inference
* VLA-Cache inference acceleration

This is important because the project is about efficiency comparison. If every method uses a different evaluation script or different input data, the comparison will not be fair.

The baseline evaluation should first focus on a small fixed subset. This makes debugging faster and cheaper. After the pipeline is stable, the same design can be scaled to more samples or the full dataset.

## 5.1 Action Unnormalization Key

The action unnormalization key should not be assumed before the LIBERO dataset structure and OpenVLA model metadata are checked.

In the previous DF-02-03 smoke test, `bridge_orig` was used because the initial OpenVLA inference example follows the BridgeData V2 style action unnormalization setup. However, DF-03-01 uses the modified LIBERO RLDS dataset, so the correct action unnormalization setting must be verified before running the structured baseline evaluation.

For DF-03-01, the unnormalization key is marked as `TBD`. It should be confirmed in DF-03-02 by checking:

* the downloaded LIBERO RLDS dataset structure
* the available OpenVLA dataset statistics
* the model or processor metadata
* the expected action dimension and value range
* whether a LIBERO-specific fine-tuned checkpoint provides its own action statistics

If no LIBERO-specific unnormalization key is available for the first inspection run, the result should be clearly marked as a pipeline/debugging run, not as a final policy-quality evaluation.

## 6. Input Format

Each evaluation sample should contain at least:

* image observation
* language instruction
* optional ground-truth action
* dataset split
* task name or episode id if available

The OpenVLA model input should be built from:

* image
* instruction text

If the dataset provides ground-truth actions, they can be used later for action-level comparison. If ground-truth actions are not used in the first baseline run, the evaluation should still record model output format, action parsing status, latency, and GPU memory usage.

## 6.1 Evaluation Split and Initial Sample Count

The local modified LIBERO RLDS dataset check confirmed the available split as:

`train`

No separate `validation` or `val` split was confirmed during the current check.

Therefore, the baseline evaluation should use a fixed subset from the available `train` split. The subset should be selected with a fixed random seed so that the same samples can be reused for baseline, LoRA, QLoRA, quantized inference, and VLA-Cache comparison.

Planned setting:

* source split: `train`
* random seed: `42`
* debug subset: `20 episodes`
* first formal subset: `50 episodes`
* medium subset: `200 episodes`
* final full evaluation: `1693 episodes`

The 20-episode run should be treated as a debugging run. The first formal baseline CSV should start from 50 fixed episodes.

Latency should not be measured from a single run only. Each sample should use one warmup run first, followed by multiple measured runs. The first planned setting is:

* warmup runs: `1`
* measured runs per sample: `5`

The final latency value should be reported as the average latency across the measured runs. This makes the latency result more stable than a single measurement.

## 6.2 Updated Evaluation Scale Plan

The evaluation should not start directly from the full dataset. The project should use a staged evaluation plan to reduce debugging cost.

The planned evaluation scale is:

| Stage      | Evaluation size | Purpose                                                                             |
| ---------- | --------------: | ----------------------------------------------------------------------------------- |
| Debug run  |     20 episodes | check model loading, data loading, output format, latency logging, and VRAM logging |
| Small run  |     50 episodes | first structured baseline result and repeatability check                            |
| Medium run |    200 episodes | stronger comparison between baseline, LoRA, QLoRA, and quantized inference          |
| Full run   |   1693 episodes | final full-dataset evaluation if the pipeline is stable                             |

The full 1693-episode evaluation is now considered feasible because the confirmed dataset size is smaller than the earlier rough assumption. However, full evaluation should only be run after the 20, 50, and 200 episode stages work correctly.

## 7. Output Metrics

The baseline evaluation should record the following metrics:

| Metric                 | Meaning                                                        |
| ---------------------- | -------------------------------------------------------------- |
| sample_id              | unique sample number                                           |
| dataset_name           | dataset used for the sample                                    |
| dataset_split          | dataset split used for evaluation                              |
| task_name              | task name if available                                         |
| episode_id             | episode id if available                                        |
| instruction            | language instruction                                           |
| image_path_or_id       | image file path or dataset id                                  |
| action_raw             | raw model output before parsing                                |
| action_parsed          | parsed action vector                                           |
| action_parse_success   | whether the output can be converted into a valid action vector |
| action_parse_error     | error reason if parsing fails                                  |
| latency_ms             | inference latency in milliseconds                              |
| peak_vram_allocated_gb | peak allocated GPU memory                                      |
| peak_vram_reserved_gb  | peak reserved GPU memory                                       |
| dtype                  | model dtype, for example bfloat16                              |
| device                 | GPU device used                                                |
| model_id               | model checkpoint name                                          |
| notes                  | extra notes or error message                                   |

## 7.1 Definition of Action Parse Success

`action_parse_success` means that the OpenVLA output can be converted into the expected continuous robot action vector format.

A sample should be marked as `action_parse_success = true` only if all of the following conditions are met:

* the model returns an output instead of an empty result
* the output can be parsed into a numerical action vector
* the parsed action vector has the expected action dimension
* all action values are finite numbers, not `NaN` or `Inf`
* the action values are within the expected valid range after any required normalization or clipping check

A sample should be marked as `action_parse_success = false` if one of the following cases happens:

* the model produces no usable output
* the output cannot be converted into a numerical action vector
* the action dimension is wrong
* the action contains `NaN` or `Inf`
* the action values are clearly outside the expected range
* an exception happens during model inference or action parsing

This metric does not measure whether the robot task succeeds. It only checks whether the model output is valid enough to be used as a robot action command. Real robot or simulator task success should be treated as a separate metric in later experiments.

## 8. Planned CSV Output

The first baseline CSV should be generated from a fixed 50-episode evaluation subset. The same subset should be reused when comparing baseline, LoRA, QLoRA, quantized inference, and VLA-Cache results.

The planned initial evaluation setting is:

* evaluation split: fixed subset from `train`
* number of samples: `50 episodes`
* random seed: `42`
* warmup runs: `1`
* measured runs per sample: `5`
* latency report: average latency over measured runs
* hardware information: recorded in the Markdown log
* package versions: recorded in the Markdown log

The planned CSV result file for future baseline evaluation is:

`results/tables/df_03_02_libero_baseline_metrics.csv`

DF-03-01 only defines the plan. The actual CSV result will be generated in DF-03-02 or later.

## 9. Planned Log Output

The planned Markdown log file for future baseline evaluation is:

`logs/df_03_02_libero_baseline_eval_log.md`

The log should include:

* cloud machine information
* GPU model
* Python version
* key package versions
* dataset path
* dataset split
* number of samples tested
* number of warmup runs
* number of measured runs
* average latency
* peak VRAM
* action parse success rate
* action unnormalization key status
* known limitations

## 10. Dataset Download Check

After the dataset download finishes on the cloud server, a separate download check log should be created.

Planned file:

`logs/df_03_01_dataset_download_check.md`

This file should record:

* dataset path
* dataset size
* number of files found
* whether incomplete files exist
* package versions
* download status
* local sub-dataset paths
* confirmed episode count
* conservative step-count planning assumption

Confirmed dataset size:

`9.6G`

Confirmed episode count:

`1693 episodes`

Confirmed local sub-datasets:

* `libero_10_no_noops`
* `libero_goal_no_noops`
* `libero_object_no_noops`
* `libero_spatial_no_noops`

## 11. Runtime Planning

The confirmed dataset contains 1693 episodes. A sampled check on `libero_goal_no_noops` showed an average trajectory length of around 118 steps per episode.

For conservative planning, the project assumes:

* 150–180 steps per episode
* approximately 250,000–300,000 total trajectory steps

The full evaluation is possible, but it should still be treated carefully because failed full runs can waste several GPU hours.

Estimated full experiment time:

| GPU level          | Estimated total time |
| ------------------ | -------------------: |
| RTX PRO 6000 level |          30–65 hours |
| H800 level         |          16–35 hours |

These are rough estimates for planning only. The final runtime should be measured and reported from actual experiment logs.

## 12. Limitations

This stage still has some limitations.

First, this task only defines the evaluation design. It does not prove final model performance.

Second, using a smaller dataset reduces engineering risk, but it may not fully represent the difficulty of real-world robot manipulation.

Third, action parse success is not the same as robot task success. A parsed action only means the model output format is valid. It does not prove that the action will succeed on a real robot or simulator.

Fourth, latency and VRAM results depend on the GPU type. Results from RTX 5090, RTX PRO 6000, A100, H800, or H100 should not be mixed as if they are directly comparable unless the hardware is clearly reported.

Fifth, the correct action unnormalization key for LIBERO must be verified before treating the model output as a final policy result. Using `bridge_orig` without confirmation may produce action values that do not match the LIBERO action distribution.

Sixth, latency measured with a small number of runs can still be noisy. The first formal baseline uses 5 measured runs per sample to reduce this issue, but the result should still be reported together with the hardware type, dtype, and batch setting.

Seventh, the estimated total trajectory steps are based on a sampled check and conservative planning assumption, not a complete full-dataset step scan.

Eighth, the current baseline evaluation plan starts from a fixed subset of the `train` split because no separate validation split was confirmed. This should be clearly reported when comparing results.

## 13. Next Task

The next task after DF-03-01 is DF-03-02.

DF-03-02 should focus on:

* loading a small number of modified LIBERO RLDS samples
* confirming the available sample fields
* confirming the correct action unnormalization key or marking it as unavailable
* converting samples into OpenVLA-compatible input
* running baseline inference on 20 episodes for debugging
* running baseline inference on 50 fixed episodes for the first formal structured result
* saving CSV and Markdown evaluation results
* keeping the same fixed subset for later LoRA, QLoRA, quantized inference, and VLA-Cache comparison
