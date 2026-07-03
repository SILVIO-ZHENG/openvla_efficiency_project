# DF-04-01 LoRA Training Log

## Notes

The following paths are local paths on the author's cloud machine.
They are recorded for experiment reproducibility and do not contain access credentials.

## Status

- status: training_finished
- failure_reason: 
- returncode: 0

## Required Training Fields

- training_time_hours: 46.6366
- total_training_steps: 200000
- final_train_loss: 0.309302
- best_val_loss: None
- best_checkpoint_step: None
- gradient_norm: 3.7245857237893585
- train_loss_curve_points: 400
- val_loss_curve_points: 0
- learning_rate: 0.0005
- batch_size: 16
- epochs: None
- trainable_params: 110828288
- lora_rank: 32
- lora_alpha: 16
- adapter_path: /root/autodl-tmp/openvla_efficiency_project/adapters/df_04_01_lora_full_train_seed42_rank32
- metadata_json: /root/autodl-tmp/openvla_efficiency_project/results/metadata/df_04_01_lora_training_metadata.json

## Dataset Split

- dataset_name: modified_libero_all_no_noops
- split_json: /root/autodl-tmp/openvla_efficiency_project/results/splits/df_03_02_libero_episode_split_seed42.json
- train_count: 1353
- val_count: 167
- test_count: 173

## GPU Info

- cuda_available: True
- gpu_name: NVIDIA RTX PRO 6000 Blackwell Server Edition
- gpu_count: 1
- torch_version: 2.11.0+cu128
- cuda_version: 12.8
- peak_vram_allocated_gb: 0.0
- peak_vram_reserved_gb: 0.0

## Command

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 /root/autodl-tmp/openvla/vla-scripts/finetune.py --vla_path /root/autodl-tmp/hf_cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f --data_root_dir /root/autodl-tmp/datasets/openvla_modified_libero_rlds --dataset_name modified_libero_all_no_noops --run_root_dir /root/autodl-tmp/openvla_efficiency_project/adapters/df_04_01_lora_full_train_seed42_rank32 --adapter_tmp_dir /root/autodl-tmp/openvla_efficiency_project/adapters/df_04_01_lora_full_train_seed42_rank32/adapter_tmp --batch_size 16 --max_steps 200000 --save_steps 5000 --learning_rate 0.0005 --grad_accumulation_steps 1 --image_aug True --shuffle_buffer_size 100000 --save_latest_checkpoint_only True --use_lora True --lora_rank 32 --lora_dropout 0.0 --use_quantization False --wandb_project openvla_efficiency --run_id_note df_04_01_lora_full_train_seed42_rank32
```

## Training stdout log

- /root/autodl-tmp/openvla_efficiency_project/logs/df_04_01_lora_training_log.stdout.log
