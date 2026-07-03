# DF-03-02 LIBERO Dataset Inspection Log

## Notes

The following paths are local paths on the author's cloud machine.
They are recorded for experiment reproducibility and do not contain access credentials.

## Purpose

This log records the structure inspection of the OpenVLA modified LIBERO RLDS dataset.

This task does not run OpenVLA inference and does not train any model.

## Inspection Environment

- Inspection time: 2026-06-11T00:22:25
- Python executable: `/root/miniconda3/envs/openvla310/bin/python`
- Dataset root: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds`
- Split inspected: `train`

## Summary

- Total sub-datasets inspected: 4
- Total train episodes: 1693
- Total TFRecord files: 96
- Total incomplete files: 0

## Sub-dataset Summary

| Sub-dataset | Split names | Train episodes | TFRecord files | Incomplete files | First episode read |
|---|---|---:|---:|---:|---|
| libero_10_no_noops | train | 379 | 32 | 0 | True |
| libero_goal_no_noops | train | 428 | 16 | 0 | True |
| libero_object_no_noops | train | 454 | 32 | 0 | True |
| libero_spatial_no_noops | train | 432 | 16 | 0 | True |

## First Episode Preview

### libero_10_no_noops

- Version dir: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_10_no_noops/1.0.0`
- Episode keys: `episode_metadata, episode_metadata.file_path, steps`
- Step keys: `action, discount, is_first, is_last, is_terminal, language_instruction, observation, observation.image, observation.joint_state, observation.state, observation.wrist_image, reward`
- Observation keys: `image, joint_state, state, wrist_image`
- Action summary: `shape=(7,), dtype=<dtype: 'float32'>`
- Language candidates: `step.language_instruction: shape=(), dtype=<dtype: 'string'>`
- Error: ``

### libero_goal_no_noops

- Version dir: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_goal_no_noops/1.0.0`
- Episode keys: `episode_metadata, episode_metadata.file_path, steps`
- Step keys: `action, discount, is_first, is_last, is_terminal, language_instruction, observation, observation.image, observation.joint_state, observation.state, observation.wrist_image, reward`
- Observation keys: `image, joint_state, state, wrist_image`
- Action summary: `shape=(7,), dtype=<dtype: 'float32'>`
- Language candidates: `step.language_instruction: shape=(), dtype=<dtype: 'string'>`
- Error: ``

### libero_object_no_noops

- Version dir: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_object_no_noops/1.0.0`
- Episode keys: `episode_metadata, episode_metadata.file_path, steps`
- Step keys: `action, discount, is_first, is_last, is_terminal, language_instruction, observation, observation.image, observation.joint_state, observation.state, observation.wrist_image, reward`
- Observation keys: `image, joint_state, state, wrist_image`
- Action summary: `shape=(7,), dtype=<dtype: 'float32'>`
- Language candidates: `step.language_instruction: shape=(), dtype=<dtype: 'string'>`
- Error: ``

### libero_spatial_no_noops

- Version dir: `/root/autodl-tmp/datasets/openvla_modified_libero_rlds/libero_spatial_no_noops/1.0.0`
- Episode keys: `episode_metadata, episode_metadata.file_path, steps`
- Step keys: `action, discount, is_first, is_last, is_terminal, language_instruction, observation, observation.image, observation.joint_state, observation.state, observation.wrist_image, reward`
- Observation keys: `image, joint_state, state, wrist_image`
- Action summary: `shape=(7,), dtype=<dtype: 'float32'>`
- Language candidates: `step.language_instruction: shape=(), dtype=<dtype: 'string'>`
- Error: ``

## Interpretation

The inspection confirms the available dataset fields before building the formal baseline runner.

The next step is to build a sample loader that converts image observations and language instructions into OpenVLA-compatible inputs.

Download completeness status: OK. No incomplete files were found.
