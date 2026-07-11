DF-04-02 LoRA Training Report
1. Experiment Overview
Run ID: df_04_02_lora_libero10_seed42_rank32_bs12_100k
This experiment fine-tuned OpenVLA-7B on the LIBERO-10 benchmark using parameter-efficient LoRA adaptation.
The training objective was token-level cross entropy for one-step action-token prediction.
2. Model Configuration
Base model: openvla/openvla-7b
Precision: torch.bfloat16
Device: cuda:0
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
3. Dataset
Dataset: libero_10_no_noops
Original episodes: 379
Train: 330
Validation: 10
Test: 39
Seed: 42
4. Input Configuration
Two-camera input
Wrist image: enabled
Proprioception: enabled
Proprio dimension: 8
5. LoRA Configuration
Rank: 32
Alpha: 16
Dropout: 0.0
Initialization: gaussian
Trainable parameters:
127,646,464
Total parameters:
7,668,883,648
6. Training Configuration
Batch size: 12
Learning rate: 0.0005
Optimizer: AdamW
Scheduler: MultiStepLR
Maximum steps: 100000
7. Best Validation Result
Best checkpoint:
Step 10000
Best validation loss:
2.978267
Validation token accuracy:
0.4848
Training loss at best checkpoint:
2.711969
Validation batches:
239
8. Summary
The model achieved its best validation performance at step 10000.
Training used BF16 precision, LoRA Rank 32, and token cross-entropy supervision.
The experiment used LIBERO-10 with a fixed seed-42 split (330/10/39).