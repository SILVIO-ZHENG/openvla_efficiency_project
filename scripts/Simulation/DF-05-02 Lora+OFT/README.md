# DF-05-02 LoRA+OFT LIBERO-10 Simulation

I use this project to evaluate the DF-04-03 LoRA+OFT best adapter in closed-loop LIBERO-10 simulation. The formal evaluation is complete: the policy succeeded in **470 of 500 rollouts (94.0%)**.

## Final result

I ran 50 trials for each of the 10 LIBERO-10 tasks, using the official deterministic initial states and a maximum of 520 policy-controlled steps per rollout.

| Metric | Result |
| --- | ---: |
| Completed tasks | 10/10 |
| Completed rollouts | 500/500 |
| Successful rollouts | 470 |
| Failed rollouts | 30 |
| Overall success rate | **94.0%** |
| Macro-average task success rate | **94.0%** |
| Timeout rate | 6.0% |
| Average successful-rollout length | 252.38 steps |
| Median successful-rollout length | 240 steps |
| Successful-rollout range | 151–505 steps |
| Average length across all rollouts | 268.44 steps |
| Formal runtime | 1.689 hours |
| Run completed | Yes |

All 30 failures reached the 520-step limit and were recorded as timeouts. No rollout failed because of an invalid action, NaN/Inf value, model-loading error, or simulation exception.

## Per-task results

| Task | Instruction | Success | Rate | Timeouts | Average steps to success |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Put both the alphabet soup and the tomato sauce in the basket | 49/50 | 98% | 1 | 281.7 |
| 1 | Put both the cream cheese box and the butter in the basket | 49/50 | 98% | 1 | 250.3 |
| 2 | Turn on the stove and put the moka pot on it | 48/50 | 96% | 2 | 242.6 |
| 3 | Put the black bowl in the bottom drawer of the cabinet and close it | 48/50 | 96% | 2 | 229.4 |
| 4 | Put the white mug on the left plate and the yellow-and-white mug on the right plate | 48/50 | 96% | 2 | 227.4 |
| 5 | Pick up the book and place it in the back compartment of the caddy | 44/50 | 88% | 6 | 184.2 |
| 6 | Put the white mug on the plate and the chocolate pudding to the right of the plate | 44/50 | 88% | 6 | 224.8 |
| 7 | Put both the alphabet soup and the cream cheese box in the basket | 50/50 | **100%** | 0 | 243.1 |
| 8 | Put both moka pots on the stove | 41/50 | **82%** | 9 | 396.9 |
| 9 | Put the yellow-and-white mug in the microwave and close it | 49/50 | 98% | 1 | 256.1 |

Task 7 achieved 100% success. Task 8 was the most difficult task: it produced 9 of the 30 failures and required an average of 396.9 steps even when successful. Tasks 5, 6, and 8 together account for 21 of the 30 timeouts (70%), so the remaining errors are concentrated in a small set of task-dependent, longer-horizon behaviors rather than distributed uniformly across the benchmark.

The structured records identify every failure as a timeout, but they do not assign manual behavioral labels. I therefore do not claim whether an individual failure was caused by grasping, placement, object interaction, or recovery behavior without inspecting and annotating its video.

## Model and training contract

| Field | Value |
| --- | --- |
| Training run | `df_04_03_lora_oft_libero10_seed42_rank32_bs12_100k` |
| Base model | `openvla/openvla-7b` |
| Best checkpoint | Step 100,000 |
| Best validation loss | 0.10628223688418949 |
| Training objective | Continuous L1 regression |
| Action head | `L1RegressionActionHead` |
| Policy output | One continuous `[8, 7]` action chunk per inference |
| Inputs | Primary image, wrist image, normalized 8-D proprio |
| Dataset | `libero_10_no_noops` |
| Fixed split | Train 330, validation 10, test 39; seed 42 |
| Normalization | Train-only `BOUNDS_Q99` statistics |
| LoRA | Rank 32, alpha 16, dropout 0 |
| Inference dtype | BF16 |
| Attention implementation | Eager |
| Diffusion / FiLM | Disabled / disabled |

The policy requires the saved LoRA adapter, processor, `action_head.pt`, `proprio_projector.pt`, dataset statistics, runtime training configuration, and training summary. It treats a missing continuous action head as a hard error and never falls back to discrete action-token prediction.

## Observation and action contract

For each model query, I provide:

```text
agentview image
+ robot0 eye-in-hand image
+ EEF position [3]
+ EEF quaternion converted to axis-angle [3]
+ gripper qpos [2]
= two images + proprio [8]
```

Both camera images are rotated by 180 degrees and processed using the configured JPEG round trip, Lanczos resize, and deterministic center crop with area scale 0.9. Proprio is normalized using the train-only q01/q99 bounds and clipped to `[-1, 1]`.

The continuous action head returns eight seven-dimensional actions. The simulator executes them in order and checks success after every individual action. A new observation is collected after each environment step, while the next model query occurs only after the current action chunk is exhausted. If success occurs inside a chunk, execution stops immediately; consequently, 136,016 actions were generated but only 134,219 were executed.

## Evaluation protocol

| Field | Value |
| --- | --- |
| Suite | LIBERO-10 |
| Tasks | 0–9 |
| Trials per task | 50 |
| Total rollouts | 500 |
| Experiment seed | 42 |
| Environment seed | 0 |
| Initial states | Official deterministic initial states |
| Initial settling | 10 dummy steps |
| Maximum controlled steps | 520 |
| Model query interval | Once per eight executed actions |
| Success check | After every environment action |
| Early stopping | Immediately on success |
| Action clipping | `[-1, 1]` |
| Gripper conversion | Official OpenVLA-OFT-to-LIBERO convention, then binarization |

This is chunked closed-loop control: the environment remains closed loop across chunks, while the eight actions inside a chunk are executed open loop.

## Inference performance on RTX 5090

The formal run used an NVIDIA GeForce RTX 5090 with 31.36 GiB of reported device memory.

| Metric | Result |
| --- | ---: |
| Total model queries | 17,002 |
| Total actions generated | 136,016 |
| Total actions executed | 134,219 |
| Average inference latency per chunk | 135.889 ms |
| P95 inference latency per chunk | 148.845 ms |
| Average amortized inference latency per executed action | 17.214 ms |
| Average environment-step latency | 15.614 ms |
| P95 environment-step latency | 22.122 ms |
| Average end-to-end latency per chunk | 294.480 ms |
| P95 end-to-end latency per chunk | 351.642 ms |
| Average amortized end-to-end latency per action | 37.303 ms |
| Effective control frequency | 26.808 Hz |
| Peak allocated VRAM | 15.193 GiB |
| Peak reserved VRAM | 15.508 GiB |

Because one inference produces eight actions, I use amortized per-action latency and effective control frequency for comparisons with one-step policies. Per-query latency alone is not a fair efficiency comparison against DF-05-01.

## Center-collapse diagnosis

I retained non-intervening diagnostics specifically to test whether the center/zero-action collapse observed in DF-05-01 reappeared during LoRA+OFT control.

| Diagnostic across the formal run | Count | Rate |
| --- | ---: | ---: |
| Zero-equivalent arm actions | 0 | 0.0% |
| Zero-equivalent full no-ops | 0 | 0.0% |
| Repeated actions | 0 | 0.0% |
| Repeated action chunks | 0 | 0.0% |
| Collapse warnings | 0 | — |
| Rollouts with a collapse warning | 0 | 0.0% |
| Invalid actions | 0 | 0.0% |

The diagnostics observed all 134,219 executed actions and did not modify, replace, reject, or terminate any model action. `collapse_action_intervention_enabled` remained `false` throughout the run.

I therefore conclude that the specific discrete zero-token/no-op collapse found in DF-05-01 did **not** occur in DF-05-02 under the recorded diagnostic definition. This does not mean that a continuous action can never be small; it means that no executed action satisfied the predefined physical zero-bin condition, no complete no-op was detected, and no repeated-action failure pattern was recorded.

The successful 94% result also shows that the LIBERO environment and observation pipeline can support effective closed-loop control. The earlier DF-05-01 failure should therefore not be attributed to the simulator alone; it was tied to the discrete one-step policy/checkpoint path that is absent here.

## Verified runtime environment

```text
GPU: NVIDIA GeForce RTX 5090
Python: 3.10.20
PyTorch: 2.11.0+cu128
CUDA: 12.8
Transformers: 4.40.1
PEFT: 0.11.1
TensorFlow: 2.15.1
OpenCV: 4.11.0.86
LIBERO: 0.1.0
Robosuite: 1.4.0
MuJoCo: 3.3.2
```

The DF-04-03 training record reports an RTX PRO 6000 Blackwell Server Edition; that is training provenance, not the simulation GPU.

The formal run started at `2026-07-12T17:24:00Z` and finished at `2026-07-12T19:05:49Z`. I invoke `/root/miniconda3/envs/openvla310/bin/python` explicitly so that shell activation state cannot select a different interpreter.

### Git provenance note

Cloud branch enforcement was intentionally disabled. The runtime manifest records `git_branch=DF-04-01`, although the resolved experiment configuration identifies the run as `DF-05-02`. I therefore use the run ID, task ID, resolved paths, checkpoint step, and recorded artifact SHA-256 values as the authoritative experiment identity; I do not use the cloud branch label as evidence of the simulation configuration.

## Project structure

```text
DF-05-02 Lora+OFT/
├── config/
│   └── simulation_config.yaml
├── launch/
│   ├── run_simulation.ps1
│   └── run_simulation.sh
├── output/
│   ├── logs/
│   ├── metrics/
│   ├── videos/
│   └── smoke_*/
├── src/
│   ├── __init__.py
│   ├── libero_env.py
│   ├── lora_oft_policy.py
│   └── simulation_runner.py
├── .gitignore
├── README.md
└── run_simulation.py
```

The downloaded `output` contains 551 files and occupies approximately 0.73 GiB. It includes 500 formal rollout videos and 12 staged smoke-test videos.

## Output records

| File | Purpose |
| --- | --- |
| `output/logs/full_console.log` | Complete launcher and terminal output |
| `output/logs/simulation.log` | High-level rollout progress |
| `output/metrics/simulation_summary.json` | Overall success, timing, VRAM, and collapse summary |
| `output/metrics/task_summary.csv` | Per-task aggregate results |
| `output/metrics/rollout_results.csv` | One row for each of the 500 formal rollouts |
| `output/metrics/failure_cases.csv` | The 30 timeout cases and diagnostic details |
| `output/metrics/inference_events.jsonl` | Per-query chunks, timings, executed actions, and diagnostic masks |
| `output/metrics/runtime_manifest.json` | Runtime versions, model paths, hashes, and policy contract |
| `output/metrics/resolved_config.yaml` | Exact resolved formal evaluation configuration |
| `output/videos/task_XX_trial_YYY.mp4` | One video for each formal rollout |

The formal metrics directory is the source of truth for the results in this README. Smoke-test results remain in separate subdirectories and are not included in the 500-rollout statistics.

## Reproducing the evaluation

The default cloud project path is:

```text
/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-05-02 Lora+OFT
```

I validate the configuration with:

```bash
cd "/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-05-02 Lora+OFT"
/root/miniconda3/envs/openvla310/bin/python run_simulation.py \
  --config config/simulation_config.yaml \
  --validate-only
```