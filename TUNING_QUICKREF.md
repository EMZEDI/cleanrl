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

## Generic Tuning (any env + script)
```bash
# Usage:
# sbatch benchmark/tune_generic.sh <env_id> <search_space> <script_path> [timesteps] [num_trials]

sbatch benchmark/tune_generic.sh Humanoid-v4 ppo cleanrl/ppo_humanoid_sparse.py 50000000 500
sbatch benchmark/tune_generic.sh Humanoid-v4 dart cleanrl/dart_humanoid_sparse_opt.py 50000000 500
sbatch benchmark/tune_generic.sh Humanoid-v4 ppo_large_critic cleanrl/ppo_humanoid_sparse_large_critic.py 50000000 500

# Pass extra fixed args (example disables torch_compile):
# EXTRA_ARGS="--fixed-arg torch-compile=False" sbatch benchmark/tune_generic.sh ...
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

# Optional: completed vs pruned vs failed counts (Optuna TrialState enums)****
sqlite3 /scratch/shahradm/optuna_humanoid.db \
  "SELECT s.study_name, t.state, COUNT(*) AS n FROM trials t JOIN studies s ON s.study_id=t.study_id GROUP BY s.study_name, t.state ORDER BY s.study_name, t.state;"

# If you see a low RUNNING count with SQLite, that's usually lock contention.
# The tuner defaults to a per-node Optuna coordinator (only SLURM_NNODES Optuna/DB clients total) to keep GPUs busy.
```

## Analyze Results
```bash
# Compare both algorithms
python cleanrl_utils/analyze_optuna_results.py --compare

# Analyze or compare arbitrary studies
python cleanrl_utils/analyze_optuna_results.py --study-name ppo_Humanoid-v4_50M
python cleanrl_utils/analyze_optuna_results.py --study-names ppo_Humanoid-v4_50M,dart_Humanoid-v4_50M --labels PPO,DART

# Individual analysis
python cleanrl_utils/analyze_optuna_results.py --algorithm ppo
python cleanrl_utils/analyze_optuna_results.py --algorithm dart
```

## Final Validation (30 runs, ~2.5 hours)
```bash
sbatch benchmark/final_comparison.sh
```

## Generic Final Validation (20 seeds)
```bash
# sbatch benchmark/final_eval_generic.sh <env_id> <script_path> <config_dir> [timesteps] [num_seeds]
sbatch benchmark/final_eval_generic.sh Humanoid-v4 cleanrl/ppo_humanoid_sparse.py /scratch/shahradm/optuna_results/ppo_Humanoid-v4_50M 50000000 20
```

## One-command Pipeline (tune → analyze → final eval)
```bash
# bash benchmark/pipeline_submit.sh <env_id> <ppo_script> <dart_script> <ppo_large_script> [timesteps] [num_trials]
bash benchmark/pipeline_submit.sh Humanoid-v4 \
  cleanrl/ppo_humanoid_sparse.py \
  cleanrl/dart_humanoid_sparse_opt.py \
  cleanrl/ppo_humanoid_sparse_large_critic.py \
  50000000 500
```

## Key Locations
- Database: `/scratch/shahradm/optuna_humanoid.db`
- Results: `/scratch/shahradm/optuna_results/`
- Logs: `/scratch/shahradm/slurm_logs/`
- Best configs: `/scratch/shahradm/optuna_results/{ppo,dart}/`
- Final eval summaries: `/scratch/shahradm/final_eval_results/`

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
