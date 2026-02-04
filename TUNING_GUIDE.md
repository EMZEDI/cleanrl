# Massive-Scale Hyperparameter Tuning (Generic)

Distributed Optuna-based hyperparameter optimization designed for large multi-node runs. This guide supports any env + script path, plus PPO, DART, and PPO large-critic variants.

## Quick Start

### 1. Launch Hyperparameter Tuning (3-hour burst)

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

### 2. Analyze Results

```bash
# After jobs complete, on head node with Python environment

# Analyze a study
python cleanrl_utils/analyze_optuna_results.py --study-name ppo_Humanoid-v4_50M

# Compare multiple studies
python cleanrl_utils/analyze_optuna_results.py --study-names ppo_Humanoid-v4_50M,dart_Humanoid-v4_50M --labels PPO,DART
```

**Outputs:**
- Top-10 trial summaries printed to console
- Best configurations saved to `/scratch/shahradm/optuna_results/{study_name}/`
- Plots: optimization history, parameter importances, PPO vs DART comparison

### 3. Run Final Validation (Top configs × 20 seeds each)

```bash
# Run final evaluation for a single study (top 3 configs × 20 seeds)
sbatch benchmark/final_eval_generic.sh Humanoid-v4 cleanrl/ppo_humanoid_sparse.py /scratch/shahradm/optuna_results/ppo_Humanoid-v4_50M 50000000 20

# Evaluate end models (default 10 episodes) and save JSON summaries
# Override with EVAL_EPISODES as needed:
# EVAL_EPISODES=20 sbatch benchmark/final_eval_generic.sh ...
### 4. One-command Pipeline (tune → analyze → final eval)

```bash
bash benchmark/pipeline_submit.sh Humanoid-v4 \
    cleanrl/ppo_humanoid_sparse.py \
    cleanrl/dart_humanoid_sparse_opt.py \
    cleanrl/ppo_humanoid_sparse_large_critic.py \
    50000000 500
```
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
4. **One Optuna coordinator per node** talks to SQLite; child GPU workers do not
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

### Test on Single Node (4 GPUs)

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
tail -f /scratch/shahradm/slurm_logs/tune_humanoid_*.out

# Count trials per study
sqlite3 /scratch/shahradm/optuna_humanoid.db \
    "SELECT s.study_name, COUNT(*) FROM trials t JOIN studies s ON s.study_id=t.study_id GROUP BY s.study_name;"
```

### Run Smaller-Scale Tuning

```bash
# If you want even more trials, just resubmit!
# Optuna handles continuation automatically
sbatch benchmark/tune_generic.sh Humanoid-v4 ppo cleanrl/ppo_humanoid_sparse.py 50000000 500
# ... wait 3 hours ...
sbatch benchmark/tune_generic.sh Humanoid-v4 ppo cleanrl/ppo_humanoid_sparse.py 50000000 500

# Or use fewer nodes for longer allocation (if preferred)
sbatch --nodes=20 --time=12:00:00 benchmark/tune_generic.sh Humanoid-v4 ppo cleanrl/ppo_humanoid_sparse.py 50000000 500
```

**Pro tip:** Multiple 3-hour jobs > One 12-hour job because:
- Instant scheduling vs. waiting in queue
- Higher worker count yields more trial starts per burst window
- Optuna picks up where you left off automatically

### Tune Only Critical Parameters

Edit [cleanrl_utils/tune_ppo_dart_humanoid.py](cleanrl_utils/tune_ppo_dart_humanoid.py):

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

### SQLite Lock Contention

If many workers cause database lock issues, switch to PostgreSQL:

```bash
# On head node, start PostgreSQL (if available)
module load postgresql
pg_ctl -D /scratch/shahradm/optuna_pg_data start

# Update storage URL in scripts
STORAGE="postgresql://user:pass@headnode:5432/optuna_humanoid"
```

### Out of Memory

Reduce batch size in trials:
```bash
# Edit tune_ppo_dart_humanoid.py, change fixed_params:
"num-envs": 32,  # Instead of 64
```

### Slow Filesystem I/O

Use local node storage for tensorboard logs:
```bash
# In tune_ppo_dart_humanoid.py, before running trial:
export TMPDIR=/tmp/node_local
# Then rsync results back to /scratch at end
```

### Failed Nodes

Optuna automatically handles node failures. Failed trials are marked and new workers pick up remaining trials. To manually retry failed trials:

```python
import optuna

study = optuna.load_study(
    study_name="ppo_Humanoid-v4_50M",
    storage="sqlite:////scratch/shahradm/optuna_humanoid.db"
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

- **Storage:** `/scratch/shahradm/optuna_humanoid.db`
- **Output:** `/scratch/shahradm/optuna_results/`
- **Logs:** `/scratch/shahradm/slurm_logs/`
- **Runs:** `runs/{experiment_name}/`

**Environment:** Scripts use `source .env` to load uv and your Python environment.

Make sure these directories exist:
```bash
mkdir -p /scratch/shahradm/{optuna_results,slurm_logs}
```
