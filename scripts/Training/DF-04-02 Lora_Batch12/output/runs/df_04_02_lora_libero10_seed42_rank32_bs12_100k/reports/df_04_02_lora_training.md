# DF-04-02 LoRA Training Log

## Status

- status: training_finished
- failure_reason:
- returncode: 0

## Model Source

- model_hub_id: openvla/openvla-7b
- requested_model_local_path: /root/autodl-tmp/hf_cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
- actual_model_loaded_from: /root/autodl-tmp/hf_cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
- actual_local_files_only: True

## Actual LoRA Runtime Configuration

- actual_lora_rank: 32
- actual_lora_alpha: 16
- actual_lora_dropout: 0.0
- actual_target_modules: ['down_proj', 'fc1', 'fc2', 'fc3', 'gate_proj', 'k_proj', 'kv', 'lm_head', 'o_proj', 'proj', 'q', 'q_proj', 'qkv', 'up_proj', 'v_proj']
- actual_init_lora_weights: gaussian
- trainable_params: 110828288
- total_params: 7652065472

## Training and Validation

- training_time_hours: 39.16
- total_training_steps: 100000
- final_train_loss: 1.36
- best_val_loss: 2.9782669634001024
- best_checkpoint_step: 10000
- validation_start_step: 10000
- validation_interval_steps: 10000
- best_model_start_step: 10000
- overfitting_checked: True
- overfitting_detected: True
- overfitting_reason: validation_loss_increased_after_best_checkpoint

## Dataset and Normalization

- dataset_name: libero_10_no_noops
- split_json: /root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/inputs/splits/libero_10_episode_split_seed42.json
- train_count: 330
- val_count: 10
- test_count: 39
- normalization_stats_json: /root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/inputs/model_config/dataset_statistics_libero_10_train_seed42.json
- normalization_stats_source_requested: train_only
- normalization_stats_source_actual: train_only

## Outputs

- best_adapter_path: /root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/output/runs/df_04_02_lora_libero10_seed42_rank32_bs12_100k/models/best_adapter
- final_adapter_path: /root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/output/runs/df_04_02_lora_libero10_seed42_rank32_bs12_100k/models/final_adapter
- resume_checkpoints: []
- runtime_resolved_config_json: /root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/output/runs/df_04_02_lora_libero10_seed42_rank32_bs12_100k/manifests/runtime_resolved_config.json
- training_summary_json: /root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/output/runs/df_04_02_lora_libero10_seed42_rank32_bs12_100k/metrics/training_summary.json
- validation_history_jsonl: /root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/output/runs/df_04_02_lora_libero10_seed42_rank32_bs12_100k/metrics/validation_history.jsonl

## Command

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
'/root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/scripts/finetune.py' \
--vla_path /root/autodl-tmp/hf_cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f \
--model_hub_id openvla/openvla-7b \
--local_files_only True \
--torch_dtype bfloat16 \
--device cuda \
--data_root_dir /root/autodl-tmp/datasets/openvla_modified_libero_rlds \
--dataset_name libero_10_no_noops \
--episode_split_json '/root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/inputs/splits/libero_10_episode_split_seed42.json' \
--train_split_name train \
--val_split_name val \
--test_split_name test \
--expected_train_episodes 330 \
--expected_val_episodes 10 \
--expected_test_episodes 39 \
--split_seed 42 \
--normalization_stats_json '/root/autodl-tmp/openvla_efficiency_project/scripts/DF-04-02 Lora_Batch12/inputs/model_config/dataset_statistics_libero_10_train_seed42.json' \
--normalization_stats_source train_only \
--allow_pipeline_debug_stats False \
--training_objective token_cross_entropy \
--use_l1_regression False \
--use_action_head False \
--use_diffusion False \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--batch_size 12 \
--max_steps 100000 \
--learning_rate 0.0005 \
--grad_accumulation_steps 1 \
--image_aug True \
--use_quantization False \
--random_seed 42 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 16 \
--lora_dropout 0.0 \
--lora_target_modules all-linear \
--init_lora_weights gaussian \
--validation_start_step 10000 \
--validation_interval_steps 10000 \
--best_model_start_step 10000
```