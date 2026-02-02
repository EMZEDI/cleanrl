# Hyperparameter Tuning: Quick Reference

## Setup (One-time)
```bash
cd /scratch/s/shahradm/cleanrl
bash setup_tuning.sh
```

## Launch Tuning (320 GPUs × 3 trials/GPU = 960 workers for 3 hours!)
```bash
# Instant allocation! Each 3-hour run completes 500+ trials
sbatch benchmark/tune_humanoid_820gpu.sh ppo    # 320 GPUs, 960 workers
sbatch benchmark/tune_humanoid_820gpu.sh dart   # 320 GPUs, 960 workers
```

**Why 3 hours works:**
- 70% of trials pruned at 20 min → 960 workers × 9 cycles = ~8,640 starts
- 30% survive to 2.5 hrs → ~2,880 completions  
- More than enough for robust tuning!

## Monitor
```bash
# Check jobs
squeue -u $USER

# Watch logs
tail -f /scratch/shahradm/slurm_logs/tune_humanoid_*.out

# Check progress
sqlite3 /scratch/shahradm/optuna_humanoid.db \
  "SELECT study_name, COUNT(*) FROM trials GROUP BY study_name;"
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
