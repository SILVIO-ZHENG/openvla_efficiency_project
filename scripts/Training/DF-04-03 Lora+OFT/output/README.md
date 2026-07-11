===== RESULT FILES =====
console_log              exists=1     size=7,175,582  console.log
training_stdout_log      exists=1     size=7,176,720  training.stdout.log
runtime_config           exists=1     size=3,981  runtime_resolved_config.json
training_summary         exists=1     size=54,662  training_summary.json
validation_history       exists=1     size=7,315  validation_history.jsonl
metadata                 exists=1     size=38,763  df_04_03_lora_oft_metadata.json
summary_csv              exists=1     size=1,468  df_04_03_lora_oft_summary.csv
markdown_log             exists=1     size=5,601  df_04_03_lora_oft_training.md

===== TRAINING SUMMARY =====
status                         = 'training_finished'
failure_reason                 = ''
training_time_hours            = 40.022911
total_training_steps           = 100000
final_train_loss               = 0.03951345384120941
best_val_loss                  = 0.10628223688418949
best_checkpoint_step           = 100000
gradient_norm                  = 8.20364444406408
peak_vram_allocated_gb         = 78.495267
peak_vram_reserved_gb          = 79.091797

===== VALIDATION HISTORY =====
validation_rows = 10
expected_steps  = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
actual_steps    = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

===== LAST VALIDATION =====
step                           = 100000
validation_type                = 'full'
validation_completed           = True
validation_stop_reason         = 'dataloader_exhausted'
train_loss                     = 0.03951345384120941
val_loss                       = 0.10628223688418949
curr_action_l1_loss            = 0.10637827107772296
next_actions_l1_loss           = 0.10626851356543568
val_batches_count              = 229
validation_time_seconds        = 130.872916

===== ACTUAL RUNTIME CONFIGURATION =====
gpu_name                     = NVIDIA RTX PRO 6000 Blackwell Server Edition
torch_dtype                  = torch.bfloat16
device                       = cuda:0
lora_rank                    = 32
lora_alpha                   = 16
lora_dropout                 = 0.0
target_modules               = ['down_proj', 'fc1', 'fc2', 'fc3', 'gate_proj', 'k_proj', 'kv', 'lm_head', 'o_proj', 'proj', 'q', 'q_proj', 'qkv', 'up_proj', 'v_proj']
init_lora_weights            = gaussian
per_device_batch_size        = 12
effective_global_batch_size  = 12
max_steps                    = 100000
trainable_params             = 278764295
total_params                 = 7820001479
image_aug                    = True
split_totals                 = {'original_episode_count': 379, 'train_count': 330, 'val_count': 10, 'test_count': 39}
num_images_in_input          = 2
use_proprio                  = True
action_dim                   = 7
proprio_dim                  = 8
num_actions_chunk            = 8

===== MODEL BUNDLES =====

[best_adapter]
directory = outputs/adapters/df_04_03_lora_oft_libero10_seed42_rank32_bs12_100k/best_adapter
adapter_config.json                        exists=1     size=1,017
adapter_model.safetensors                  exists=1     size=484,458,600
action_head.pt                             exists=1     size=302,242,379
proprio_projector.pt                       exists=1     size=67,275,329
dataset_statistics.json                    exists=1     size=3,248
dataset_statistics_provenance.json         exists=1     size=4,748
validation_metrics.json                    exists=1     size=955
processor/                                 exists=True files=6

[final_adapter]
directory = outputs/adapters/df_04_03_lora_oft_libero10_seed42_rank32_bs12_100k/final_adapter
adapter_config.json                        exists=1     size=1,017
adapter_model.safetensors                  exists=1     size=484,458,600
action_head.pt                             exists=1     size=302,242,379
proprio_projector.pt                       exists=1     size=67,275,329
dataset_statistics.json                    exists=1     size=3,248
dataset_statistics_provenance.json         exists=1     size=4,748
validation_metrics.json                    exists=1     size=145
processor/                                 exists=True files=6

===== CURVE LENGTHS =====
train_loss_curve               = 200
val_loss_curve                 = 10
learning_rate_curve            = 200
gradient_norm_curve            = 200
validation_history             = 10

DF_04_03_GENERATED_ARTIFACTS_AUDIT_PASSED