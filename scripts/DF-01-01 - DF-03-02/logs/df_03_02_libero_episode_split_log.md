# DF-03-02 LIBERO Episode Split Log

## Notes

The following paths are local paths on the author's cloud machine.
They are recorded for experiment reproducibility and do not contain access credentials.

## Purpose

This log records the fixed per-package 80/10/10 episode split for the OpenVLA modified LIBERO RLDS dataset.

This task does not run OpenVLA inference and does not train any model.

This task only reads dataset_info.json and saves lightweight split metadata.

## Split Settings

- Created at: `2026-06-11T00:44:13`
- Dataset name: `openvla/modified_libero_rlds`
- Dataset root: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds`
- Source split: `train`
- Random seed: `42`
- Split scope: `per_sub_dataset`
- Physical data copy: `False`
- Train ratio: `0.8`
- Validation ratio: `0.1`
- Test ratio: `0.1`
- Split JSON path: `results/splits/df_03_02_libero_episode_split_seed42.json`
- Split rule: Each LIBERO sub-dataset is split independently. For each sub-dataset, episode indices are shuffled with a deterministic package-specific seed. train_count=int(total*0.8), val_count=int(total*0.1), and test receives the remaining episodes.

## Original Dataset Paths

| Sub-dataset | Task suite | Version directory |
|---|---|---|
| libero_spatial_no_noops | spatial | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_spatial_no_noops/1.0.0` |
| libero_object_no_noops | object | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_object_no_noops/1.0.0` |
| libero_goal_no_noops | goal | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_goal_no_noops/1.0.0` |
| libero_10_no_noops | libero_10 | `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_10_no_noops/1.0.0` |

## Split Count Summary

| Sub-dataset | Task suite | Original episodes | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| libero_spatial_no_noops | spatial | 432 | 345 | 43 | 44 |
| libero_object_no_noops | object | 454 | 363 | 45 | 46 |
| libero_goal_no_noops | goal | 428 | 342 | 42 | 44 |
| libero_10_no_noops | libero_10 | 379 | 303 | 37 | 39 |
| **Total** | **-** | **1693** | **1353** | **167** | **173** |

## How This Split Will Be Used

The generated JSON file is the single source of truth for later dataset loading.

The training loader should read `train_episode_indices` from each sub-dataset record.

The validation loader should read `val_episode_indices` from each sub-dataset record.

The test loader should read `test_episode_indices` only for final evaluation.

For DF-03-02 baseline open-loop inference, a fixed 50-sample evaluation subset should be selected from the test split, not from the train or validation split.

## Important Git Rule

The generated JSON and Markdown log are lightweight metadata files and can be committed to GitHub.

The real dataset files, TFRecord shards, model weights, cache folders, and credentials must not be committed.

## Interpretation

This split is a logical episode-level split by LIBERO sub-dataset. The original dataset folders remain unchanged.

This means the project does not create physical train/val/test dataset copies. It only records which episode indices should be used for each split.
