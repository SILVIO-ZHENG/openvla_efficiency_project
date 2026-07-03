DF-04-03 LoRA+OFT Training Log

In DF-04-03, I completed the main LoRA+OFT training run for the OpenVLA efficiency project.
The experiment used the `libero_10_no_noops` dataset and a fixed split with seed 42:
330 training episodes
10 validation episodes
39 test episodes
The model was based on OpenVLA-7B and trained on one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU using BF16 precision.
The training setup was:
LoRA rank: 32
LoRA alpha: 16
LoRA dropout: 0.0
LoRA target modules: all linear layers
Batch size: 12
Gradient accumulation: 1
Effective batch size: 12
Learning rate: 0.0005
Maximum training steps: 100,000
Image augmentation: enabled for training
Validation interval: every 10,000 steps
This experiment used LoRA together with the OFT-style continuous action head.
The model took:
the main camera image
the wrist camera image
the language instruction
an 8-dimensional proprioceptive state
It predicted an action chunk with shape:

[batch, 8, 7]

This means that each sample produced 8 future actions, with 7 continuous action values per step.
The training objective was continuous L1 action regression rather than action-token cross-entropy.
The trainable parts were:
LoRA parameters inside the OpenVLA backbone
the continuous action head
the proprioception projector
The final runtime configuration reported:
Total parameters: 7,820,001,479
Trainable parameters: 278,764,295
This means that around 3.56% of the full model parameters were trainable.
---
Training result
The training finished successfully.

status = training_finished
total_training_steps = 100000
training_time_hours = 40.022911

The final recorded values were:

final_train_loss = 0.03951345384120941
best_val_loss = 0.10628223688418949
best_checkpoint_step = 100000
gradient_norm = 8.20364444406408

The best validation result was reached at the final training step.
At step 100,000, the validation result was:

val_loss = 0.10628223688418949
curr_action_l1_loss = 0.10637827107772296
next_actions_l1_loss = 0.10626851356543568
val_batches = 229
validation_time_seconds = 130.872916
overfitting_detected = False
best_improved = True

The current-action loss and future-action loss were very close, so the model did not show a clear difference between predicting the immediate action and predicting the rest of the action chunk.
The validation process completed normally with:
 
validation_completed = True
validation_stop_reason = dataloader_exhausted
  
The training script ran validation 10 times:
 
10000
20000
30000
40000
50000
60000
70000
80000
90000
100000
  
The final validation was also the best one.
---
Training behaviour
At step 500, the recorded values were approximately:
 
train_loss = 0.298575
gradient_norm = 15.946977
  
At step 100,000, they were:
 
train_loss = 0.039513
gradient_norm = 8.203644
  
So the training loss dropped substantially during training, and the gradient norm also became smaller overall.
The training log contained:
200 training-loss records
200 learning-rate records
200 gradient-norm records
10 validation records
The logging interval was every 500 steps.
---
GPU and training cost
The run used:
 
GPU = NVIDIA RTX PRO 6000 Blackwell Server Edition
dtype = torch.bfloat16
device = cuda:0
  
The peak memory usage was:
 
peak_vram_allocated_gb = 78.495267
peak_vram_reserved_gb = 79.091797
  
The whole run took about 40 hours.
The average wall-clock time was about 1.44 seconds per training step, including validation, logging, and model saving.
---
Saved models
The run produced two model folders:
 
best_adapter/
final_adapter/
  
Each model folder contains:
 
README.md
adapter_config.json
adapter_model.safetensors
action_head.pt
proprio_projector.pt
dataset_statistics.json
dataset_statistics_provenance.json
validation_metrics.json
processor/
  
The main model files were:
 
adapter_model.safetensors = 484,458,600 bytes
action_head.pt = 302,242,379 bytes
proprio_projector.pt = 67,275,329 bytes
  
The processor folder contains the tokenizer and image preprocessing files needed to load the model correctly.
Because the best checkpoint was step 100,000, both `best_adapter` and `final_adapter` were saved from the final stage of training.
---
Files produced by the run
The completed run produced 40 files in total.
They were organised locally into:
 
output/
└── runs/
    └── df_04_03_lora_oft_libero10_seed42_rank32_bs12_100k/
        ├── models/
        │   ├── best_adapter/
        │   └── final_adapter/
        ├── metrics/
        ├── logs/
        ├── reports/
        ├── provenance/
        └── manifests/
  
The main generated files include:
Models
 
models/best_adapter/
models/final_adapter/
  
Metrics
 
runtime_resolved_config.json
training_summary.json
validation_history.jsonl
  
Logs
 
console.log
training.stdout.log
  
Reports
 
df_04_03_lora_oft_metadata.json
df_04_03_lora_oft_summary.csv
df_04_03_lora_oft_training.md
  
Run provenance
 
launch_resolved_config.yaml
dataset_statistics.json
dataset_statistics_provenance.json
episode_split.json
  
These files record the actual runtime settings, data split, normalisation statistics, training curves, validation history, saved models, and full console output.
---
Warning at shutdown
At the end of the run, PyTorch printed this warning:
 
destroy_process_group() was not called before program exit
  
This happened after the model files and result files had already been saved.
The wrapper still finished with:
 
LoRA+OFT wrapper finished with status: training_finished
  
So the warning did not affect the saved model or the recorded results. It was related to NCCL process-group cleanup when the program exited.
---
DF-04-03 result
DF-04-03 was completed successfully.
The main outcome was a trained OpenVLA LoRA+OFT model with:
 
100,000 training steps
40.02 hours of training
0.039513 final training L1 loss
0.106282 best validation L1 loss
best checkpoint at step 100,000
78.50 GB peak allocated VRAM
79.09 GB peak reserved VRAM
  
The model, logs, metrics, runtime configuration, dataset split, and training reports were all saved successfully.