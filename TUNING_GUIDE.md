# Massive-Scale Hyperparameter Tuning (Generic)

Distributed Optuna-based hyperparameter optimization designed for large multi-node runs. This guide supports any env + script path, plus PPO, DART, and PPO large-critic variants.

## Quick Start

### 1. Setup Database (Run Once)

Before running any tuning, initialize the PostgreSQL database on the head node:

```bash
# Initialize and start PostgreSQL
bash benchmark/setup_postgres.sh
```

### 2. Launch Hyperparameter Tuning (3-hour burst)

Syntax:
```bash
sbatch benchmark/tune_generic.sh <env_id> <search_space> <script_path> <total_timesteps> <num_trials>
```

Example (Humanoid-v4):

```bash
# In your cleanrl fork directory, with .env available

# Tune PPO
sbatch benchmark/tune_generic.sh Humanoid-v4 ppo cleanrl/ppo_humanoid_sparse.py 50000000 500

# Tune DART
sbatch benchmark/tune_generic.sh Humanoid-v4 dart cleanrl/dart_humanoid_sparse_opt.py 50000000 500

# Tune PPO large-critic
sbatch benchmark/tune_generic.sh Humanoid-v4 ppo_large_critic cleanrl/ppo_humanoid_sparse_large_critic.py 50000000 500
```

**Expected results:**
- High parallelism for fast trial starts in short allocations
- Trials per GPU configured via `TRIALS_PER_GPU` (default 3)
- Pruning is currently seed-step based (not timestep based) unless you change it

### 3. Analyze Results

```bash
# After jobs complete, on head node with Python environment

# Analyze a study
# Syntax: python cleanrl_utils/analyze_optuna_results.py --study-name <study_name>
python cleanrl_utils/analyze_optuna_results.py --study-name ppo_Humanoid-v4_50M

# Compare multiple studies
python cleanrl_utils/analyze_optuna_results.py --study-names ppo_Humanoid-v4_50M,dart_Humanoid-v4_50M --labels PPO,DART
```

**Outputs:**
- Top-10 trial summaries printed to console
- Best configurations saved to `/scratch/shahradm/optuna_results/{study_name}/`
- Plots: optimization history, parameter importances, PPO vs DART comparison

### 4. Run Final Validation (Top configs × 20 seeds each)

```bash
# Syntax: sbatch benchmark/final_eval_generic.sh <env_id> <script_path> <results_dir> <timesteps> <seeds>
# Run final evaluation for a single study (top 3 configs × 20 seeds)
sbatch benchmark/final_eval_generic.sh Humanoid-v4 cleanrl/ppo_humanoid_sparse.py /scratch/shahradm/optuna_results/ppo_Humanoid-v4_50M 50000000 20

# Evaluate end models (default 10 episodes) and save JSON summaries
# Override with EVAL_EPISODES as needed:
# EVAL_EPISODES=20 sbatch benchmark/final_eval_generic.sh ...
```

### 5. One-command Pipeline (tune → analyze → final eval)

```bash
bash benchmark/pipeline_submit.sh <env_id> \
    <ppo_script_path> \
    <dart_script_path> \
    <ppo_large_critic_script_path> \
    <total_timesteps> <num_trials>
```

Example:

```bash
bash benchmark/pipeline_submit.sh Humanoid-v4 \
    cleanrl/ppo_humanoid_sparse.py \
    cleanrl/dart_humanoid_sparse_opt.py \
    cleanrl/ppo_humanoid_sparse_large_critic.py \
    50000000 500
```

**Expected results:**
- 30 runs complete in ~2.5 hours on 4 GPUs
- Results logged to Wandb project `humanoid_ppo_vs_dart_final`
- Model checkpoints and videos saved

---

## Architecture

### Files Created

```
cleanrl_utils/
    └── tune_generic.py             # Generic tuner (per-node Optuna coordinator)
    └── final_eval_generic.py       # Final eval + end-model evaluation
    └── analyze_optuna_results.py   # Analysis & visualization

benchmark/
    └── tune_generic.sh             # SLURM launcher for generic tuning
    └── final_eval_generic.sh       # SLURM launcher for final eval
    └── pipeline_submit.sh          # Tune → analyze → final eval pipeline
```

### How It Works

1. **SLURM job** requests N nodes × 4 GPUs
2. Each node sources `.env` to load uv and your Python environment
3. Each node runs [cleanrl_utils/tune_generic.py](cleanrl_utils/tune_generic.py) via [benchmark/tune_generic.sh](benchmark/tune_generic.sh)
4. **One Optuna coordinator per node** talks to PostgreSQL; child GPU workers do not
5. **Trials per GPU** keeps each GPU busy (default 3)
6. Each trial runs `total-timesteps` with your fixed args and tuned hyperparameters

### Hyperparameter Search Spaces

**PPO:**
- `learning-rate`: 1e-5 to 1e-3 (log scale)
- `gae-lambda`: 0.92 to 0.99
- `ent-coef`: 0.001 to 0.02 (log scale)
- `clip-coef`: 0.1 to 0.3
- `num-minibatches`: {4, 8, 16}
- `update-epochs`: {4, 8, 10}

**DART:**
- `learning-rate`: 1e-5 to 1e-3 (log scale)
- `dart-lr-scale`: 0.1 to 0.3
- `dart-lambda-res`: 0.997 to 0.9999
- `dart-warmup-frac`: 0.3 to 0.5
- `num-minibatches`: {4, 8, 16}
- `update-epochs`: {4, 8, 10}

**Fixed for speed (defaults):**
- `num-envs`: 64
- `num-steps`: 1024
- `total-timesteps`: 50M

You can add extra fixed args via `EXTRA_ARGS="--fixed-arg key=value"` when launching.

---

## Value-Function Analysis Metrics (Wandb)

The humanoid PPO/DART scripts now log a unified value-analysis namespace under `analysis/*`.

### Core metrics (compare PPO vs DART directly)

- `analysis/mse_v_total`: Empirical value error $\mathbb{E}[(V(s)-G^{MC})^2]$ for the active critic.
    - Lower is better.
- `analysis/mse_v_base`: Base critic error against MC return.
    - For PPO, this equals `analysis/mse_v_total` (single critic).
- `analysis/mse_improvement`: `mse_v_base - mse_v_total`.
    - Positive means DART residual is improving value fit.
- `analysis/advantage_correlation`: $\mathrm{Corr}(\hat A_t, A_t^{true})$ where $A_t^{true}\approx G_t^{MC}-V(s_t)$.
    - Higher is better; indicates training advantages align with MC-based advantages.
- `analysis/mean_v_total`, `analysis/mean_mc_return`:
    - Track calibration/drift of value scale vs return scale.

### DART-only diagnostics

- `dart/mean_base_value`: Mean of base critic predictions.
- `dart/mean_residual_value`: Mean residual contribution.
    - Positive residual suggests underestimation correction by base critic.
    - Negative residual suggests overestimation correction by base critic.

### How these guide tuning decisions

1. If `analysis/mse_v_total` is flat/high for both PPO and DART:
     - Reduce `learning-rate`, increase `update-epochs`, or increase `num-minibatches`.
2. If DART has weak/negative `analysis/mse_improvement`:
     - Increase `dart-lr-scale` slightly or reduce `dart-warmup-frac`.
     - If unstable, do the opposite (lower `dart-lr-scale`, later warmup).
3. If `analysis/advantage_correlation` is low/noisy:
     - Increase rollout quality (more timesteps/envs), tune `gae-lambda`, or reduce policy update aggressiveness.
4. If `analysis/mean_v_total` diverges far from `analysis/mean_mc_return`:
     - Value function is miscalibrated; tune value-related settings first (`vf-coef`, LR, clipping behavior).

> Note: legacy tags under `value_metrics/*` are still logged for backward compatibility.

---

## Resource Estimates

### Per Trial
- **Time:** ~2.5 hours (50M timesteps @ 5000-6000 SPS)
- **GPU:** 1 × L40s
- **CPU:** 4 cores
- **Memory:** ~8 GB

### Full Tuning Run (500 trials/algorithm)
- **Wall time:** depends on `total-timesteps` and cluster scheduling
- **GPU-hours:** proportional to completed trials

### Final Validation (top 3 configs × 20 seeds)
- **Wall time:** depends on `total-timesteps` and GPU count

---

## Advanced Usage

```bash
# Interactive test on 1 node
srun --nodes=1 --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=4:00:00 --pty bash

cd /scratch/shahradm/cleanrl

# Run 20 trials across 4 GPUs
python cleanrl_utils/tune_generic.py \
    --script-path cleanrl/ppo_humanoid_sparse.py \
    --env-id Humanoid-v4 \
    --search-space ppo \
    --num-trials 20 \
    --num-gpus 4
```

### Monitor Progress

```bash
# Check SLURM job status
squeue -u $USER

# Tail logs from specific node
tail -f /scratch/shahradm/slurm_logs/tune_generic_*.out

# Count trials per study
psql -U optuna -d optuna_humanoid -h vulcan1 -c \
    "SELECT s.study_name, COUNT(*) FROM trials t JOIN studies s ON s.study_id=t.study_id GROUP BY s.study_name;"
```

### Run Smaller-Scale Tuning

```bash
# If you want even more trials, just resubmit!
# Optuna handles continuation automatically
sbatch benchmark/tune_generic.sh <env_id> <search_space> <script_path> <timesteps> <trials>

# Or use fewer nodes for longer allocation (if preferred)
sbatch --nodes=20 --time=12:00:00 benchmark/tune_generic.sh <env_id> <search_space> <script_path> <timesteps> <trials>
```

**Pro tip:** Multiple 3-hour jobs > One 12-hour job because:
- Instant scheduling vs. waiting in queue
- Higher worker count yields more trial starts per burst window
- Optuna picks up where you left off automatically

### Tune Only Critical Parameters

Edit [cleanrl_utils/tune_generic.py](cleanrl_utils/tune_generic.py):

```python
# Simplified PPO search (learning rate + lambda only)
def get_ppo_params(trial: optuna.Trial) -> Dict[str, any]:
    return {
        "learning-rate": trial.suggest_float("learning-rate", 1e-5, 1e-3, log=True),
        "gae-lambda": trial.suggest_float("gae-lambda", 0.92, 0.99),
        # Remove other params
    }
```

### Adjust Trials Per GPU

If you experience OOM errors or want to tune memory/compute tradeoff:

```bash
# Edit benchmark/tune_generic.sh
TRIALS_PER_GPU=2  # More conservative (2 trials per GPU)
TRIALS_PER_GPU=4  # More aggressive (4 trials per GPU, ~10GB each)
```

Memory usage per trial: ~8GB (32 envs) to ~15GB (128 envs)

---

## Troubleshooting

### Database Issues

If the PostgreSQL server stops (e.g., after a node reboot), restart it:

```bash
# Check status
pg_ctl -D /scratch/shahradm/optuna_pg_data status

# Start if stopped
pg_ctl -D /scratch/shahradm/optuna_pg_data -l /scratch/shahradm/optuna_pg.log start
```

### Out of Memory

Reduce batch size in trials:
```bash
# Increase fixed_arg when launching:
sbatch benchmark/tune_generic.sh ... EXTRA_ARGS="--fixed-arg num-envs=32"
```

### Slow Filesystem I/O

Use local node storage for tensorboard logs:
```bash
# In your training script (e.g. ppo.py), support a --tmp-dir arg or similar:
sbatch benchmark/tune_generic.sh ... EXTRA_ARGS="--fixed-arg tmp-dir=/tmp/node_local"
```

### Failed Nodes

Optuna automatically handles node failures. Failed trials are marked and new workers pick up remaining trials. To manually retry failed trials:

```python
import optuna

study = optuna.load_study(
    study_name="ppo_Humanoid-v4_50M",
    storage="postgresql://optuna:optuna_secure_pwd_2026@vulcan1:5432/optuna_humanoid"
)

# Get failed trials
failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
print(f"{len(failed)} failed trials")

# Optuna will automatically sample new trials to replace them
```

---

## Next Steps After Tuning

1. **Upload results to Wandb** (from head node with internet):
   ```bash
   # Parse Optuna DB and upload to wandb
   python cleanrl_utils/upload_results_to_wandb.py
   ```

2. **Run extended validation** (100M-150M timesteps):
   ```bash
   # Use best config for publication-quality results
   sbatch benchmark/extended_validation.sh
   ```

3. **Ablation studies**:
   - Fix best learning rate, sweep other params
   - Test on different sparse reward configs
   - Compare on Humanoid-v5 or other MuJoCo tasks

---

## Configuration Files

All scripts use these defaults (can override via command-line):

- **Storage:** `postgresql://optuna:optuna_secure_pwd_2026@vulcan1:5432/optuna_humanoid`
- **Output:** `/scratch/shahradm/optuna_results/`
- **Logs:** `/scratch/shahradm/slurm_logs/`
- **Runs:** `runs/{experiment_name}/`

**Environment:** Scripts use `source .env` to load uv and your Python environment.

Make sure these directories exist:
```bash
mkdir -p /scratch/shahradm/{optuna_results,slurm_logs}
```
