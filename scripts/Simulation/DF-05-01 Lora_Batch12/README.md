# DF-05-01 LoRA Batch12 LIBERO Simulation

This project evaluates the DF-04-02 LoRA-OneStepCE best adapter in closed-loop LIBERO-10 simulation.

## Confirmed model contract

- Training run: `df_04_02_lora_libero10_seed42_rank32_bs12_100k`
- Best checkpoint: step 10,000
- Objective: token cross-entropy
- Policy: one-step discrete action tokens
- Output: one `[7]` action per inference
- Inputs: primary image, wrist image, and normalized 8-D proprio
- Dataset statistics key: `libero_10_no_noops`
- Dtype: BF16

The training data uses `future_action_window_size=0` and seven action tokens. The bundled OpenVLA source still contains the old LIBERO `NUM_ACTIONS_CHUNK=8` inference constant. `src/lora_policy.py` explicitly sets that runtime value to `1` and rejects any output that is not shaped `[1, 7]`.

## Directory structure

```text
DF-05-01 Lora_Batch12/
├── config/simulation_config.yaml
├── launch/run_simulation.ps1
├── launch/run_simulation.sh
├── output/
│   ├── logs/
│   ├── metrics/
│   └── videos/
├── src/
│   ├── __init__.py
│   ├── libero_env.py
│   ├── lora_policy.py
│   └── simulation_runner.py
├── .gitignore
├── README.md
└── run_simulation.py
```

## Verified cloud environment

The working compatibility combination is:

```text
Python 3.10.20
PyTorch 2.11.0+cu128
Transformers 4.40.1
MuJoCo 3.3.2
Robosuite 1.4.0
LIBERO 0.1.0
BDDL 1.0.1
Gym 0.25.2
PyOpenGL 3.1.10
```

MuJoCo 3.10 is not compatible with Robosuite 1.4.0 because its `mj_fullM` binding changed. Robosuite 1.5 is not compatible with this LIBERO checkout because `SingleArmEnv` moved.

The system EGL loader must also exist:

```bash
apt-get update
apt-get install -y libegl1
```

Do not install the complete historical LIBERO `requirements.txt` into `openvla310`; it would downgrade OpenVLA dependencies. Install only the confirmed compatibility packages when required.

## Observation preprocessing

Each control step uses:

```text
agentview_image
robot0_eye_in_hand_image
robot0_eef_pos
robot0_eef_quat converted to axis-angle
robot0_gripper_qpos
```

Both camera images are rotated 180 degrees. Each image follows the official JPEG round-trip and Lanczos resize, followed by a deterministic center crop with area scale `0.9`. The processor produces six channels per image, and the input is channel-stacked in this order:

```text
primary [1, 6, 224, 224]
wrist   [1, 6, 224, 224]
combined [1, 12, 224, 224]
```

Raw proprio is constructed as:

```text
EEF position [3] + EEF axis-angle [3] + gripper qpos [2] = [8]
```

It is normalized with the train-only `q01` and `q99` values and clipped to `[-1, 1]`.

## Evaluation protocol

- Suite: LIBERO-10
- Tasks: 10
- Formal rollouts per task: 50
- Total formal rollouts: 500
- Environment seed: 0
- Fixed official initial states
- Initial settling steps: 10
- Maximum policy-controlled steps: 520
- Stop immediately when LIBERO returns success
- One fresh model inference for every environment action

For staged debugging, edit only `simulation_config.yaml`:

```yaml
simulation:
  task_ids: [0]
  num_tasks: 1
  num_trials_per_task: 1
```

Recommended progression:

```text
1 task x 1 rollout
1 task x 5 rollouts
10 tasks x 1 rollout
10 tasks x 50 rollouts
```

## Running on the cloud

The project expects this cloud path:

```text
/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-05-01 Lora_Batch12
```

Run:

```bash
cd "/root/autodl-tmp/openvla_efficiency_project/scripts/Simulation/DF-05-01 Lora_Batch12"
bash launch/run_simulation.sh
```

The launcher activates `openvla310`, configures EGL, adds the patched DF-04-02 source and LIBERO repository to `PYTHONPATH`, and forces offline model loading.

Local Windows validation:

```powershell
cd "D:\PROJECT\Master's Thesis\openvla_efficiency_project\scripts\Simulation\DF-05-01 Lora_Batch12"
powershell -ExecutionPolicy Bypass -File .\launch\run_simulation.ps1
```

## Output files

```text
output/
├── logs/
│   ├── launcher.stdout.log
│   └── simulation.log
├── metrics/
│   ├── failure_cases.csv
│   ├── inference_events.jsonl
│   ├── resolved_config.yaml
│   ├── rollout_results.csv
│   ├── runtime_manifest.json
│   ├── simulation_summary.json
│   └── task_summary.csv
└── videos/
    └── task_XX_trial_YYY.mp4
```

Videos are closed-loop simulation rollout videos, not training videos. The default configuration saves both successful and failed rollouts. Each MP4 contains the primary and wrist views side by side, one frame per environment step, with task, trial, step, latency, and success overlays.

## Resume behavior

The default configuration resumes from `rollout_results.csv` and skips completed rollout IDs. Each rollout row and inference event is flushed immediately. Keep the existing metrics files when resuming the same run, and use a new `run_id` plus a new output directory for a different experiment.

## Expected storage

Structured logs and metrics are normally below 300 MB for 500 one-step rollouts. MP4 size depends on episode length and codec complexity. With both 256-pixel views stored side by side, a typical full run is expected to use roughly 2-6 GB. Reserve at least 10 GB, or 15 GB for comfortable safety margin.

## Success criteria

A formal run is complete when:

- `simulation_summary.json` reports `total_rollouts=500` and `completed=true`.
- Every configured task has 50 rows in `rollout_results.csv`.
- Every rollout has a valid termination reason.
- No action has an invalid shape or NaN/Inf value.
- Runtime manifest versions and model paths match the intended environment.

## Runtime status

The Python files and YAML can be validated locally without model weights. A real `1 task x 1 rollout` cloud smoke test is still required before starting all 500 rollouts.
