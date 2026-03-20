# Hyperparameter Tuning

Ray Tune + Optuna Bayesian optimization for YOLO on PACE/ICE cluster.

## Workflow Overview

```
1. sbatch slurm_ray_tune.sh          # Submit tuning job (20 trials, 4 GPUs)
2. ASHA early-stops bad trials        # Most trials only run 20 epochs
3. Best trials run up to 300 epochs   # max_t=300 in ASHA scheduler
4. best_hyperparams.yaml updated live # Saved after each improved trial
5. Best model saved to models/best_tune/best.pt
```

### Step 1: Submit Tuning Job

```bash
sbatch scripts/tuning/slurm_ray_tune.sh
```

### Step 2: Monitor Progress

```bash
# Job status
squeue -u $USER

# Live training output
tail -f logs/ray_tune_<job_id>.out

# Trial status table (shows all trials with metrics)
grep -A 25 "Trial name" logs/ray_tune_<job_id>.out | tail -30

# Errors (mostly harmless Ray internal warnings)
cat logs/ray_tune_<job_id>.err
```

**Ray Dashboard** (optional):

```bash
# 1. Find head node from job output
head -20 logs/ray_tune_<job_id>.out   # look for "Head Node: <hostname>"

# 2. SSH tunnel from your local machine
ssh -L 8265:<head-node-hostname>:8265 <user>@login-ice.pace.gatech.edu

# 3. Open http://localhost:8265
#    - Actors page: see running trials
#    - Cluster page: GPU/CPU usage
#    Note: Dashboard shows Ray tasks, not Tune trials. Use stdout logs for trial metrics.
```

### Step 3: Job Completes

When the job finishes, it prints a summary to stdout:

```
============================================================
TUNING COMPLETE — BEST TRIAL SUMMARY
============================================================

--- Metrics ---
  mAP50:     0.9740
  mAP50-95:  0.7375
  Precision: 0.9676
  Recall:    0.9595

--- Hyperparameters ---
  lr0: 0.000541
  batch_size: 8
  ...

--- Result Paths ---
  Ray trial dir:   <scratch>/ray_results/yolo_tune_.../train_yolo_trial_..._...
  YOLO output dir: <scratch>/tune/trial_20260207_...
    weights/best.pt — best model weights
  Copied best.pt → models/best_tune/best.pt
============================================================

All results saved to: configs/best_hyperparams.yaml
```

### Step 4: Retrain with Best Hyperparameters

The tuning job runs trials up to 300 epochs (ASHA `max_t=300`). `best_hyperparams.yaml` and `models/best_tune/best.pt` are updated live after each improved trial, so results are preserved even if the job times out.

```bash
cat configs/best_hyperparams.yaml   # Check best config
ls models/best_tune/best.pt         # Best model weights
```

## How It Works

### Search Algorithm: Optuna (Bayesian Optimization)

- Uses past trial results to predict which hyperparameters are promising
- More efficient than random search — converges faster with fewer trials
- `--num-samples 20` means 20 total hyperparameter combinations
- First trial is seeded with the previous best config (from `train` section), subsequent trials explore from there

### Scheduler: ASHA (Aggressive Early Stopping)

- `grace_period=20`: every trial runs at least 20 epochs
- `reduction_factor=3`: at each rung, keep only the top 1/3 of trials
- `max_t=300`: best trials run the full 300 epochs
- Rungs: epoch 20 → 60 → 180 → 300

This means most trials are killed quickly, saving GPU time for the promising ones.

### What Gets Tuned (Search Space)

| Category | Parameters | Range |
|----------|-----------|-------|
| Learning rate | `lr0`, `lrf` | loguniform/uniform |
| Optimizer | `momentum`, `weight_decay`, `warmup_epochs` | uniform |
| Batch size | `batch_size` | choice: [4, 8, 16] |
| Loss weights | `box`, `cls`, `dfl` | uniform |
| Augmentation | `hsv_h/s/v`, `degrees`, `translate`, `scale`, `fliplr`, `mosaic`, `mixup` | uniform |

### What is Fixed (from H5.yaml `train` section)

| Parameter | Value | Why |
|-----------|-------|-----|
| `imgsz` | 1280 | Dataset-specific, not a hyperparameter |
| `epochs` | 300 | ASHA's `max_t` controls actual training length |
| `model` | yolov11m.pt | Architecture choice is separate |
| `workers` | 2 | Matches SLURM CPU allocation |
| `patience` | 20 | Early stopping within a single trial |

## Output Locations

### During Tuning (scratch filesystem)

```
~/scratch/lizard_toepads/ray_tune/<job_id>/
├── ray_results/
│   └── yolo_tune_<timestamp>/
│       ├── train_yolo_trial_<id>_*/   # Per-trial Ray results
│       │   ├── progress.csv           # Per-epoch metrics
│       │   ├── result.json            # Final metrics
│       │   ├── params.json            # Hyperparameters
│       │   └── events.out.tfevents.*  # TensorBoard
│       └── tuner.pkl                  # Tune state (for resume)
└── tune/
    └── trial_<timestamp>/             # Per-trial YOLO outputs
        ├── weights/
        │   ├── best.pt                # Best model weights
        │   └── last.pt                # Last epoch weights
        ├── results.csv                # YOLO training log
        ├── args.yaml                  # YOLO training config
        └── *.jpg                      # Training visualizations
```

### After Tuning (project directory)

| Path | Content |
|------|---------|
| `configs/best_hyperparams.yaml` | Best hyperparameters, metrics, and paths |
| `models/best_tune/best.pt` | Best model weights (copied from trial) |
| `logs/ray_tune_<job_id>.out` | Full stdout log |
| `logs/ray_tune_<job_id>.err` | stderr (Ray warnings) |

### Symlink

`runs_scratch` → `~/scratch/lizard_toepads/ray_tune/<job_id>/` (updated each job)

## Important Notes

### SLURM Configuration

- **Do NOT use `--block`** in `ray start` — it kills the entire cluster if any subprocess (e.g. `ray_client_server`) crashes. We use `sleep infinity` instead.
- **Do NOT use `ray symmetric-run`** — it checks all users' processes on the node and fails on shared HPC nodes.
- **Port 6379** is the default Ray GCS port. On a busy cluster, another user could be using it. If `ray start` fails with address-in-use, change `PORT=6379` to another value.

### H5.yaml `tune` Section

- Do NOT hardcode `project` or `storage_path` in H5.yaml — the SLURM script sets these via environment variables (`YOLO_PROJECT`, `RAY_STORAGE_PATH`). Hardcoded paths override env vars and cause all jobs to write to the same directory.
- `search_space` overrides in H5.yaml take precedence over defaults in `tune_hyperparams.py`.

### Ray Dashboard Caveats

- Dashboard "Finished: N" counts are Ray **tasks**, not Tune trials. Check stdout for actual trial count.
- `ray[default]` must be installed for the dashboard (not just `ray[tune]`).
- Dashboard shows cluster health, not trial metrics. Use stdout or TensorBoard for metrics.

### stderr Warnings (safe to ignore)

| Warning | Cause |
|---------|-------|
| "Failed to establish connection to metrics exporter agent" | Ray internal monitoring, doesn't affect training |
| `RayDeprecationWarning: RunConfig` | Import path change, cosmetic |
| `FutureWarning: accelerator visible devices` | Future Ray behavior change |

## Quick Reference

### Job Management

```bash
sbatch scripts/tuning/slurm_ray_tune.sh   # Submit
squeue -u $USER                             # Status
scancel <job_id>                            # Cancel
```

### Check Results

```bash
# Latest trial table
grep -A 25 "Trial name" logs/ray_tune_<job_id>.out | tail -30

# Best trial after job completes
cat configs/best_hyperparams.yaml

# Best model weights
ls models/best_tune/best.pt
```

### Cluster Info

```bash
sinfo -o "%P %l %a %G"                                              # Partitions/GPUs
sacctmgr show qos coc-ice format=Name,MaxTRESMinsPerJob%60          # QOS limits
sacctmgr show assoc where user=$USER format=User,Account,QOS%30     # Your account
```

## tune_hyperparams.py Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/H5.yaml` | Base config file |
| `--num-samples` | 20 | Number of trials |
| `--max-concurrent` | auto | Max concurrent trials |
| `--gpus-per-trial` | 1.0 | GPUs per trial |
| `--cpus-per-trial` | 4 | CPUs per trial |
| `--metric` | `metrics/mAP50(B)` | Optimize metric |
| `--mode` | max | Optimization direction (max/min) |
| `--resume` | false | Resume previous run |

## QOS Limits (coc-ice)

**Constraint**: `nodes * gpus_per_node * hours <= 16`

| Config | Calculation | OK? |
|--------|------------|-----|
| 4 nodes x 1 GPU x 4h | 16 GPU-h | Yes (max) |
| 2 nodes x 1 GPU x 8h | 16 GPU-h | Yes (max) |
| 1 node x 1 GPU x 16h | 16 GPU-h | Yes (max) |

## Files

```
scripts/tuning/
  slurm_ray_tune.sh        # SLURM batch script (multi-node Ray)
  tune_hyperparams.py       # Tuning logic (Optuna + ASHA scheduler)
configs/
  H6.yaml                   # Base training config (tuned params used as seed point)
  best_hyperparams.yaml     # Output: best trial results (auto-generated)
docs/tune/
  README.md                 # This file
```
