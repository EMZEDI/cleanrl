"""
Distributed Optuna hyperparameter tuning for PPO vs DART on sparse Humanoid-v4.
Designed for massive-scale parallelization on SLURM with 820 L40s GPUs (4 per node).

Usage:
    # Launch on SLURM node with 4 GPUs (launches 4 parallel workers)
    python cleanrl_utils/tune_ppo_dart_humanoid.py --algorithm ppo --num-trials 500
    python cleanrl_utils/tune_ppo_dart_humanoid.py --algorithm dart --num-trials 500
"""

import argparse
import multiprocessing as mp
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from tensorboard.backend.event_processing import event_accumulator


def extract_metric_from_tensorboard(run_dir: str, metric: str, last_n: int = 50) -> float:
    """Extract metric from tensorboard logs."""
    try:
        ea = event_accumulator.EventAccumulator(run_dir)
        ea.Reload()
        if metric not in ea.Tags()["scalars"]:
            return float('-inf')
        metric_values = [scalar_event.value for scalar_event in ea.Scalars(metric)[-last_n:]]
        return float(np.mean(metric_values)) if metric_values else float('-inf')
    except Exception as e:
        print(f"Error extracting metric from {run_dir}: {e}")
        return float('-inf')


def get_ppo_params(trial: optuna.Trial) -> Dict[str, any]:
    """PPO hyperparameter search space."""
    return {
        "learning-rate": trial.suggest_float("learning-rate", 1e-5, 1e-3, log=True),
        "gae-lambda": trial.suggest_float("gae-lambda", 0.92, 0.99),
        "ent-coef": trial.suggest_float("ent-coef", 0.001, 0.02, log=True),
        "clip-coef": trial.suggest_float("clip-coef", 0.1, 0.3),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [4, 8, 16]),
        "update-epochs": trial.suggest_categorical("update-epochs", [4, 8, 10]),
    }


def get_dart_params(trial: optuna.Trial) -> Dict[str, any]:
    """DART hyperparameter search space."""
    return {
        "learning-rate": trial.suggest_float("learning-rate", 1e-5, 1e-3, log=True),
        "dart-lr-scale": trial.suggest_float("dart-lr-scale", 0.1, 0.3),
        "dart-lambda-res": trial.suggest_float("dart-lambda-res", 0.997, 0.9999),
        "dart-warmup-frac": trial.suggest_float("dart-warmup-frac", 0.3, 0.5),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [4, 8, 16]),
        "update-epochs": trial.suggest_categorical("update-epochs", [4, 8, 10]),
    }


def run_trial(
    algorithm: str,
    trial: optuna.Trial,
    gpu_id: int,
    seed: int = 1,
    total_timesteps: int = 50_000_000,
) -> float:
    """Run a single trial on specified GPU."""
    
    # Select script and parameters
    if algorithm == "ppo":
        script_path = "cleanrl/ppo_humanoid_sparse.py"
        params = get_ppo_params(trial)
    elif algorithm == "dart":
        script_path = "cleanrl/dart_humanoid_sparse_opt.py"
        params = get_dart_params(trial)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Fixed optimized settings for speed
    fixed_params = {
        "env-id": "Humanoid-v4",
        "total-timesteps": total_timesteps,
        "num-envs": 64,
        "num-steps": 1024,
        "seed": seed,
    }
    
    # Merge parameters
    all_params = {**fixed_params, **params}
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Build command
    sys.argv = [script_path] + [f"--{key}={value}" for key, value in all_params.items()]
    
    print(f"[GPU {gpu_id}] Trial {trial.number}: Running with params: {params}")
    
    try:
        # Run the experiment
        experiment = runpy.run_path(path_name=script_path, run_name="__main__")
        run_name = experiment.get("run_name", f"trial_{trial.number}")
        
        # Extract metric
        run_dir = f"runs/{run_name}"
        metric = "charts/episodic_return"
        avg_return = extract_metric_from_tensorboard(run_dir, metric, last_n=50)
        
        print(f"[GPU {gpu_id}] Trial {trial.number}: Average return = {avg_return:.2f}")
        
        return avg_return
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Trial {trial.number} failed: {e}")
        return float('-inf')


def create_objective(algorithm: str, gpu_id: int, total_timesteps: int, num_seeds: int):
    """Create objective function for a specific GPU worker."""
    
    def objective(trial: optuna.Trial) -> float:
        returns = []
        
        for seed in range(num_seeds):
            avg_return = run_trial(
                algorithm=algorithm,
                trial=trial,
                gpu_id=gpu_id,
                seed=seed,
                total_timesteps=total_timesteps,
            )
            
            returns.append(avg_return)
            
            # Report intermediate value for pruning (after each seed)
            trial.report(float(np.mean(returns)), step=seed)
            
            # Check if trial should be pruned
            if trial.should_prune():
                print(f"[GPU {gpu_id}] Trial {trial.number} pruned after seed {seed}")
                raise optuna.TrialPruned()
        
        # Return mean across seeds
        final_return = float(np.mean(returns))
        print(f"[GPU {gpu_id}] Trial {trial.number}: Final return = {final_return:.2f} (mean over {num_seeds} seeds)")
        
        return final_return
    
    return objective


def run_worker(
    algorithm: str,
    study_name: str,
    storage: str,
    gpu_id: int,
    n_trials: int,
    total_timesteps: int,
    num_seeds: int,
):
    """Run Optuna worker on a specific GPU."""
    print(f"[GPU {gpu_id}] Starting Optuna worker for {algorithm} (study: {study_name})")
    
    # Load or create study (with pruner and sampler)
    pruner = PercentilePruner(
        percentile=70.0,
        n_startup_trials=5,
        n_warmup_steps=3,  # Prune after 3 seeds
    )
    
    sampler = TPESampler(
        seed=42 + gpu_id,  # Different seed per worker for diversity
        n_startup_trials=10,
    )
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            pruner=pruner,
            sampler=sampler,
        )
        print(f"[GPU {gpu_id}] Loaded existing study '{study_name}'")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True,
        )
        print(f"[GPU {gpu_id}] Created new study '{study_name}'")
    
    # Create objective for this GPU
    objective = create_objective(algorithm, gpu_id, total_timesteps, num_seeds)
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"[GPU {gpu_id}] Completed {n_trials} trials for {algorithm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True, choices=["ppo", "dart"],
                        help="Algorithm to tune: ppo or dart")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Optuna study name (default: {algorithm}_humanoid_50M)")
    parser.add_argument("--storage", type=str, default="sqlite:////scratch/shahradm/optuna_humanoid.db",
                        help="Optuna storage URL")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs per node (default: 4)")
    parser.add_argument("--trials-per-gpu", type=int, default=3,
                        help="Number of concurrent trials per GPU (default: 3 for 48GB L40s)")
    parser.add_argument("--num-trials", type=int, default=500,
                        help="Total number of trials to run (split across all workers)")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000,
                        help="Training timesteps per trial")
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="Number of seeds per trial (default: 1 for speed)")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Run on single GPU instead of multi-GPU parallelization")
    
    args = parser.parse_args()
    
    # Set study name
    if args.study_name is None:
        args.study_name = f"{args.algorithm}_humanoid_{args.total_timesteps // 1_000_000}M"
    
    print("="*80)
    print(f"Optuna Distributed Tuning: {args.algorithm.upper()} on Humanoid-v4")
    print("="*80)
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Total trials: {args.num_trials}")
    print(f"GPUs per node: {args.num_gpus}")
    print(f"Trials per GPU: {args.trials_per_gpu}")
    print(f"Total workers: {args.num_gpus * args.trials_per_gpu}")
    print(f"Timesteps per trial: {args.total_timesteps:,}")
    print(f"Seeds per trial: {args.num_seeds}")
    print("="*80)
    
    # Create storage directory if using SQLite
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    if args.single_gpu:
        # Run on single GPU (useful for testing)
        run_worker(
            algorithm=args.algorithm,
            study_name=args.study_name,
            storage=args.storage,
            gpu_id=0,
            n_trials=args.num_trials,
            total_timesteps=args.total_timesteps,
            num_seeds=args.num_seeds,
        )
    else:
        # Launch parallel workers (multiple per GPU)
        total_workers = args.num_gpus * args.trials_per_gpu
        trials_per_worker = args.num_trials // total_workers
        
        print(f"Launching {total_workers} workers ({args.trials_per_gpu} per GPU)...")
        
        processes = []
        worker_id = 0
        for gpu_id in range(args.num_gpus):
            for _ in range(args.trials_per_gpu):
                p = mp.Process(
                    target=run_worker,
                    args=(
                        args.algorithm,
                        args.study_name,
                        args.storage,
                        gpu_id,
                        trials_per_worker,
                        args.total_timesteps,
                        args.num_seeds,
                    )
                )
                p.start()
                processes.append(p)
                worker_id += 1
                time.sleep(0.5)  # Stagger starts slightly
        
        print(f"All {total_workers} workers launched!")
        
        # Wait for all workers to complete
        for p in processes:
            p.join()
    
    print("="*80)
    print(f"All workers completed for {args.algorithm}")
    print(f"Results stored in: {args.storage}")
    print("="*80)


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
