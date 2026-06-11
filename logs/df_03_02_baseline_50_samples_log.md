# DF-03-02 Baseline 50-Sample Manifest Log

## Notes

The following paths are local paths on the author's cloud machine.
They are recorded for experiment reproducibility and do not contain access credentials.

## Purpose

This log records the fixed 50-episode evaluation subset for the DF-03-02 LIBERO baseline run.

This task does not run OpenVLA inference and does not train any model.

## Important Scope

This manifest selects episode indices only. It does not select concrete step indices or image frames.

The later baseline runner should read each selected episode, choose one fixed step, and then use `observation.image` plus `language_instruction` for OpenVLA baseline inference.

## Sample Selection Settings

- Dataset name: `openvla/modified_libero_rlds`
- Dataset root: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds`
- Source split: `train`
- Evaluation split: `test`
- Random seed: `42`
- Number of samples: `50`
- Selection level: `episode`
- Step selection status: `not_selected_in_this_script`
- Physical data copy: `False`
- Output JSON: `results\splits\df_03_02_baseline_50_samples_seed42.json`
- Output CSV: `results\tables\df_03_02_baseline_50_samples_manifest.csv`

## Allocation Summary

| Sub-dataset | Task suite | Available test episodes | Selected samples |
|---|---|---:|---:|
| libero_spatial_no_noops | spatial | 44 | 13 |
| libero_object_no_noops | object | 46 | 13 |
| libero_goal_no_noops | goal | 44 | 13 |
| libero_10_no_noops | libero_10 | 39 | 11 |

## Selected Samples

| Sample ID | Sub-dataset | Task suite | Eval split | Episode index |
|---|---|---|---|---:|
| df_03_02_baseline_001 | libero_10_no_noops | libero_10 | test | 87 |
| df_03_02_baseline_002 | libero_10_no_noops | libero_10 | test | 93 |
| df_03_02_baseline_003 | libero_10_no_noops | libero_10 | test | 130 |
| df_03_02_baseline_004 | libero_10_no_noops | libero_10 | test | 219 |
| df_03_02_baseline_005 | libero_10_no_noops | libero_10 | test | 221 |
| df_03_02_baseline_006 | libero_10_no_noops | libero_10 | test | 225 |
| df_03_02_baseline_007 | libero_10_no_noops | libero_10 | test | 227 |
| df_03_02_baseline_008 | libero_10_no_noops | libero_10 | test | 247 |
| df_03_02_baseline_009 | libero_10_no_noops | libero_10 | test | 307 |
| df_03_02_baseline_010 | libero_10_no_noops | libero_10 | test | 317 |
| df_03_02_baseline_011 | libero_10_no_noops | libero_10 | test | 346 |
| df_03_02_baseline_012 | libero_goal_no_noops | goal | test | 38 |
| df_03_02_baseline_013 | libero_goal_no_noops | goal | test | 40 |
| df_03_02_baseline_014 | libero_goal_no_noops | goal | test | 132 |
| df_03_02_baseline_015 | libero_goal_no_noops | goal | test | 196 |
| df_03_02_baseline_016 | libero_goal_no_noops | goal | test | 216 |
| df_03_02_baseline_017 | libero_goal_no_noops | goal | test | 235 |
| df_03_02_baseline_018 | libero_goal_no_noops | goal | test | 289 |
| df_03_02_baseline_019 | libero_goal_no_noops | goal | test | 293 |
| df_03_02_baseline_020 | libero_goal_no_noops | goal | test | 304 |
| df_03_02_baseline_021 | libero_goal_no_noops | goal | test | 332 |
| df_03_02_baseline_022 | libero_goal_no_noops | goal | test | 341 |
| df_03_02_baseline_023 | libero_goal_no_noops | goal | test | 357 |
| df_03_02_baseline_024 | libero_goal_no_noops | goal | test | 400 |
| df_03_02_baseline_025 | libero_object_no_noops | object | test | 25 |
| df_03_02_baseline_026 | libero_object_no_noops | object | test | 164 |
| df_03_02_baseline_027 | libero_object_no_noops | object | test | 200 |
| df_03_02_baseline_028 | libero_object_no_noops | object | test | 262 |
| df_03_02_baseline_029 | libero_object_no_noops | object | test | 267 |
| df_03_02_baseline_030 | libero_object_no_noops | object | test | 270 |
| df_03_02_baseline_031 | libero_object_no_noops | object | test | 274 |
| df_03_02_baseline_032 | libero_object_no_noops | object | test | 289 |
| df_03_02_baseline_033 | libero_object_no_noops | object | test | 291 |
| df_03_02_baseline_034 | libero_object_no_noops | object | test | 321 |
| df_03_02_baseline_035 | libero_object_no_noops | object | test | 342 |
| df_03_02_baseline_036 | libero_object_no_noops | object | test | 352 |
| df_03_02_baseline_037 | libero_object_no_noops | object | test | 409 |
| df_03_02_baseline_038 | libero_spatial_no_noops | spatial | test | 43 |
| df_03_02_baseline_039 | libero_spatial_no_noops | spatial | test | 91 |
| df_03_02_baseline_040 | libero_spatial_no_noops | spatial | test | 112 |
| df_03_02_baseline_041 | libero_spatial_no_noops | spatial | test | 179 |
| df_03_02_baseline_042 | libero_spatial_no_noops | spatial | test | 270 |
| df_03_02_baseline_043 | libero_spatial_no_noops | spatial | test | 282 |
| df_03_02_baseline_044 | libero_spatial_no_noops | spatial | test | 291 |
| df_03_02_baseline_045 | libero_spatial_no_noops | spatial | test | 304 |
| df_03_02_baseline_046 | libero_spatial_no_noops | spatial | test | 311 |
| df_03_02_baseline_047 | libero_spatial_no_noops | spatial | test | 313 |
| df_03_02_baseline_048 | libero_spatial_no_noops | spatial | test | 323 |
| df_03_02_baseline_049 | libero_spatial_no_noops | spatial | test | 359 |
| df_03_02_baseline_050 | libero_spatial_no_noops | spatial | test | 387 |

## How This Manifest Will Be Used

The baseline runner should read this manifest and run OpenVLA inference on these fixed 50 episodes only.

The 50 selected episodes are only used to validate the DF-03-02 baseline pipeline, field recording, fixed step selection, action parsing, latency logging, and VRAM logging.

Formal method evaluation will use the full test split / full-step open-loop evaluation, not this 50-sample debug manifest.

