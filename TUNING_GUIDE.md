# Massive-Scale Hyperparameter Tuning: PPO vs DART on Humanoid-v4

Distributed Optuna-based hyperparameter optimization designed for 820 L40s GPUs (205 nodes × 4 GPUs).

## Quick Start

### 1. Launch Hyperparameter Tuning (960 parallel workers in 3-hour burst!)

```bash
# In your cleanrl fork directory, with .env available

# Tune PPO (80 nodes × 4 GPUs × 3 trials/GPU = 960 workers)
sbatch benchmark/tune_humanoid_820gpu.sh ppo

# Tune DART (80 nodes × 4 GPUs × 3 trials/GPU = 960 workers)
sbatch benchmark/tune_humanoid_820gpu.sh dart
```

**Expected results:**
- **960 parallel workers** exploiting instant 3-hour allocation
- Each GPU runs **3 concurrent trials** (~12GB each)
- **Aggressive pruning** kills 70% of trials at 5M steps (~20 min)
- **~8,640 trial starts** in 3 hours (workers cycle through trials)
- **~2,880 high-quality completions** (30% survive to 50M steps)
- **Way more than 500 trials per algorithm** in a single job!
- Job starts **instantly** vs. waiting hours for 12+ hour slots

### 2. Analyze Results

```bash
# After jobs complete, on head node with Python environment

# Analyze individual algorithms
python cleanrl_utils/analyze_optuna_results.py --algorithm ppo
python cleanrl_utils/analyze_optuna_results.py --algorithm dart

# Compare both algorithms
python cleanrl_utils/analyze_optuna_results.py --compare
```

**Outputs:**
- Top-10 trial summaries printed to console
- Best configurations saved to `/scratch/s/shahradm/optuna_results/{ppo,dart}/`
- Plots: optimization history, parameter importances, PPO vs DART comparison

### 3. Run Final Validation (Top-3 configs × 5 seeds each)

```bash
# Run 30 final experiments: (3 PPO + 3 DART) × 5 seeds
sbatch benchmark/final_comparison.sh
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
  └── tune_ppo_dart_humanoid.py   # Main tuning script (multi-GPU worker)
  └── analyze_optuna_results.py   # Analysis & visualization

benchmark/
  └── tune_humanoid_820gpu.sh     # SLURM launcher for 600 GPUs
  └── final_comparison.sh         # Final validation runs
```

### How It Works

1. **SLURM job** requests 150 nodes × 4 GPUs = 600 GPUs total
2. Each node sources `.env` to load uv and your Python environment
3. Each node runs `tune_ppo_dart_humanoid.py` which spawns **12 workers** (3 per GPU)
4. **1800 total workers** run trials concurrently, sharing GPU memory efficiently
5. Workers connect to **shared SQLite database** on `/scratch/shahradm/optuna_humanoid.db`
6. Optuna coordinates trial distribution across all 1800 workers
7. **Percentile pruner** kills bottom 70% of trials after 3 seeds (~5M steps)
8. Each trial runs 50M timesteps with optimized settings (~2.5 hours)
9. Multiple trials per GPU utilize L40s' 48GB memory (~12GB per trial)

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

**Fixed for speed:**
- `num-envs`: 64
- `num-steps`: 1024
- `vectorization`: async
- `torch-compile`: True
- `enable-tf32`: True
- `total-timesteps`: 50M

---

## Resource Estimates

### Per Trial
- **Time:** ~2.5 hours (50M timesteps @ 5000-6000 SPS)
- **GPU:** 1 × L40s
- **CPU:** 4 cores
- **Memory:** ~8 GB

### Full Tuning Run (500 trials/algorithm with 70% pruning)
- **Wall time:** 18-24 hours
- **GPU-hours:** ~1,200 per algorithm (but pruning saves ~60%)
- **Total cost:** ~500 GPU-hours per algorithm effectively

### Final Validation (30 runs)
- **Wall time:** ~2.5 hours on 4 GPUs (or 10 hours on 1 GPU)
- **GPU-hours:** 75 total

---

## Advanced Usage

### Test on Single Node (4 GPUs)

```bash
# Interactive test on 1 node
srun --nodes=1 --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=4:00:00 --pty bash

cd /scratch/s/shahradm/cleanrl

# Run 20 trials across 4 GPUs
python cleanrl_utils/tune_ppo_dart_humanoid.py \
    --algorithm ppo \
    --num-trials 20 \
    --num-gpus 4
```

### Monitor Progress

```bash
# Check SLURM job status
squeue -u $USER

# Tail logs from specific node
tail -f /scratch/shahradm/slurm_logs/tune_humanoid_*.out

# Count completed trials
sqlite3 /scratch/shahradm/optuna_humanoid.db \
    "SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE';"

# Check best value so far
sqlite3 /scratch/shahradm/optuna_humanoid.db \
    "SELECT MAX(value) FROM trials WHERE state = 'COMPLETE' AND study_id = (SELECT study_id FROM studies WHERE study_name = 'ppo_humanoid_50M');"
```

### Run Smaller-Scale Tuning

```bash
# If you want even more trials, just resubmit!
# Optuna handles continuation automatically
sbatch benchmark/tune_humanoid_820gpu.sh ppo   # Job 1: ~3K trials
# ... wait 3 hours ...
sbatch benchmark/tune_humanoid_820gpu.sh ppo   # Job 2: another ~3K trials

# Or use fewer nodes for longer allocation (if preferred)
sbatch --nodes=20 --time=12:00:00 benchmark/tune_humanoid_820gpu.sh ppo
```

**Pro tip:** Multiple 3-hour jobs > One 12-hour job because:
- Instant scheduling vs. waiting in queue
- 960 workers >> 240 workers (4× throughput)
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
# Edit benchmark/tune_humanoid_820gpu.sh
TRIALS_PER_GPU=2  # More conservative (2 trials per GPU)
TRIALS_PER_GPU=4  # More aggressive (4 trials per GPU, ~10GB each)
```

Memory usage per trial: ~8GB (32 envs) to ~15GB (128 envs)

---

## Troubleshooting

### SQLite Lock Contention

If 1800 workers cause database lock issues, switch to PostgreSQL:

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
    study_name="ppo_humanoid_50M",
    storage="sqlite:////scratch/s/shahradm/optuna_humanoid.db"
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
