# Hyperparameter Tuning: Quick Reference

## Setup (One-time)
```bash
cd /scratch/shahradm/cleanrl
bash setup_tuning.sh
```

## Launch Tuning (60 nodes × 4 GPUs × 3 trials/GPU = 720 workers for 3 hours)
```bash
# 3-hour “burst” jobs tend to schedule fast on this cluster.
# Note: current pruner is seed-step based; with NUM_SEEDS=1 most trials run full length.
sbatch benchmark/tune_humanoid_820gpu.sh ppo
sbatch benchmark/tune_humanoid_820gpu.sh dart
```

**Why 3 hours still works well:**
- Very high parallelism yields lots of *started* trials quickly.
- You can iterate allocations rapidly (submit multiple bursts) and then run final validation on top configs.

## Monitor
```bash
# Check jobs
squeue -u $USER

# Watch logs
tail -f /scratch/shahradm/slurm_logs/tune_humanoid_*.out

# 1) Confirm multi-node launch actually happened.
#    Expected: the printed count equals SLURM_NNODES (e.g., 60)
grep -m1 "srun sanity check" /scratch/shahradm/slurm_logs/tune_humanoid_*.out

# 2) Confirm expected worker fan-out.
#    New default is “per-node Optuna coordinator”, so you should see 1 coordinator per node.
#    Expected: SLURM_NNODES lines (e.g., 60)
grep -c "Coordinator starting" /scratch/shahradm/slurm_logs/tune_humanoid_*.out

# 3) Confirm trials are actually launching.
#    Expected (soon after start): up to min(NUM_TRIALS, Nnodes*4*TRIALS_PER_GPU) lines
grep -c "Running with params" /scratch/shahradm/slurm_logs/tune_humanoid_*.out

# Check progress
sqlite3 /scratch/shahradm/optuna_humanoid.db \
  "SELECT s.study_name, COUNT(*) AS n_trials FROM trials t JOIN studies s ON s.study_id=t.study_id GROUP BY s.study_name ORDER BY n_trials DESC;"

# Optional: completed vs pruned vs failed counts (Optuna TrialState enums)
sqlite3 /scratch/shahradm/optuna_humanoid.db \
  "SELECT s.study_name, t.state, COUNT(*) AS n FROM trials t JOIN studies s ON s.study_id=t.study_id GROUP BY s.study_name, t.state ORDER BY s.study_name, t.state;"

# If you see a low RUNNING count with SQLite, that's usually lock contention.
# The tuner defaults to a per-node Optuna coordinator (only SLURM_NNODES Optuna/DB clients total) to keep GPUs busy.
```

## Analyze Results
```bash
# Compare both algorithms
python cleanrl_utils/analyze_optuna_results.py --compare

# Individual analysis
python cleanrl_utils/analyze_optuna_results.py --algorithm ppo
python cleanrl_utils/analyze_optuna_results.py --algorithm dart
```

## Final Validation (30 runs, ~2.5 hours)
```bash
sbatch benchmark/final_comparison.sh
```

## Key Locations
- Database: `/scratch/shahradm/optuna_humanoid.db`
- Results: `/scratch/shahradm/optuna_results/`
- Logs: `/scratch/shahradm/slurm_logs/`
- Best configs: `/scratch/shahradm/optuna_results/{ppo,dart}/`

## Resource Usage
- **Tuning:** 600 GPUs × 20 hours = 12,000 GPU-hours (effective ~4,800 with pruning)
- **Validation:** 4 GPUs × 2.5 hours = 10 GPU-hours
- **Storage:** ~5 GB for database, ~100 GB for tensorboard logs

## Expected Results
- **PPO:** ~500 completed trials, best config in top-3
- **DART:** ~500 completed trials, best config in top-3
- **Pruned:** ~70% of trials killed early (saves compute)
- **Best returns:** 400-600+ episodic return (sparse Humanoid)

## Troubleshooting
- **Database locks:** Use PostgreSQL instead of SQLite
- **OOM:** Reduce `num-envs` from 64 to 32
- **Slow I/O:** Use node-local `/tmp` for tensorboard logs
- **Failed nodes:** Optuna auto-handles, just resubmit job

See [TUNING_GUIDE.md](TUNING_GUIDE.md) for details.
