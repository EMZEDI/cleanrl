"""
Generic distributed Optuna tuner for CleanRL-style scripts.

Supports PPO, DART, and PPO-large-critic (same space as PPO) by providing:
  - env id
  - script path
  - search-space label

Runs one Optuna coordinator per node and keeps GPUs busy with local worker
processes (children do not talk to PostgreSQL directly - only the coordinator on each node does).
"""

import argparse
import multiprocessing as mp
import os
import runpy
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import optuna
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from tensorboard.backend.event_processing import event_accumulator


def _get_slurm_world() -> tuple[int, int]:
    world_size = int(os.environ.get("SLURM_NNODES") or 1)
    rank = int(os.environ.get("SLURM_PROCID") or 0)
    return world_size, rank


def _trials_for_this_rank(total_trials: int, world_size: int, rank: int) -> int:
    base = total_trials // world_size
    rem = total_trials % world_size
    return base + (1 if rank < rem else 0)


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(2.0, 0.05 * (2**attempt)))


def extract_metric_from_tensorboard(run_dir: str, metric: str, last_n: int = 50) -> float:
    try:
        ea = event_accumulator.EventAccumulator(run_dir)
        ea.Reload()
        if metric not in ea.Tags().get("scalars", []):
            return float("-inf")
        metric_values = [scalar_event.value for scalar_event in ea.Scalars(metric)[-last_n:]]
        return float(np.mean(metric_values)) if metric_values else float("-inf")
    except Exception as e:
        print(f"Error extracting metric from {run_dir}: {e}")
        return float("-inf")


def get_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "learning-rate": trial.suggest_float("learning-rate", 1e-5, 1e-3, log=True),
        "gae-lambda": trial.suggest_float("gae-lambda", 0.92, 0.99),
        "ent-coef": trial.suggest_float("ent-coef", 0.001, 0.02, log=True),
        "clip-coef": trial.suggest_float("clip-coef", 0.1, 0.3),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [4, 8, 16]),
        "update-epochs": trial.suggest_categorical("update-epochs", [4, 8, 10]),
    }


def get_dart_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "learning-rate": trial.suggest_float("learning-rate", 1e-5, 1e-3, log=True),
        "dart-lr-scale": trial.suggest_float("dart-lr-scale", 0.1, 0.3),
        "dart-lambda-res": trial.suggest_float("dart-lambda-res", 0.997, 0.9999),
        "dart-warmup-frac": trial.suggest_float("dart-warmup-frac", 0.3, 0.5),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [4, 8, 16]),
        "update-epochs": trial.suggest_categorical("update-epochs", [4, 8, 10]),
    }


def _parse_kv_list(items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --fixed-arg '{item}', expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def run_trial_with_params(
    script_path: str,
    params: Dict[str, Any],
    trial_number: int,
    gpu_id: int,
    seed: int,
    total_timesteps: int,
    env_id: str,
    num_envs: int,
    num_steps: int,
    fixed_args: Dict[str, str],
    metric_tag: str,
    last_n: int,
) -> float:
    fixed_params = {
        "env-id": env_id,
        "total-timesteps": total_timesteps,
        "num-envs": num_envs,
        "num-steps": num_steps,
        "seed": seed,
    }
    all_params: Dict[str, Any] = {**fixed_params, **fixed_args, **params}

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.argv = [script_path] + [f"--{key}={value}" for key, value in all_params.items()]

    print(f"[GPU {gpu_id}] Trial {trial_number}: Running with params: {params}")
    try:
        experiment = runpy.run_path(path_name=script_path, run_name="__main__")
        run_name = experiment.get("run_name", f"trial_{trial_number}")
        run_dir = f"runs/{run_name}"
        avg_return = extract_metric_from_tensorboard(run_dir, metric_tag, last_n=last_n)
        print(f"[GPU {gpu_id}] Trial {trial_number}: Average return = {avg_return:.2f}")
        return avg_return
    except Exception as e:
        print(f"[GPU {gpu_id}] Trial {trial_number} failed: {e}")
        return float("-inf")


def run_node_coordinator(
    study_name: str,
    storage: str,
    script_path: str,
    search_space: str,
    env_id: str,
    num_gpus: int,
    trials_per_gpu: int,
    total_trials_job: int,
    total_timesteps: int,
    num_envs: int,
    num_steps: int,
    num_seeds: int,
    fixed_args: Dict[str, str],
    metric_tag: str,
    last_n: int,
):
    world_size, rank = _get_slurm_world()
    trials_this_node = _trials_for_this_rank(total_trials_job, world_size, rank)
    total_workers = num_gpus * trials_per_gpu

    print(
        f"[rank {rank}/{world_size}] Coordinator starting: {trials_this_node} trials on this node, "
        f"{total_workers} local workers ({trials_per_gpu} per GPU)"
    )

    pruner = PercentilePruner(percentile=70.0, n_startup_trials=5, n_warmup_steps=3)
    sampler = TPESampler(seed=42 + rank, n_startup_trials=10)

    for attempt in range(10):
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                pruner=pruner,
                sampler=sampler,
                load_if_exists=True,
            )
            break
        except Exception as e:
            print(f"[rank {rank}] Optuna create/load failed (attempt {attempt}): {e}")
            _sleep_backoff(attempt)
    else:
        raise RuntimeError("Failed to create/load Optuna study after retries")

    if search_space == "ppo" or search_space == "ppo_large_critic":
        get_params = get_ppo_params
    elif search_space == "dart":
        get_params = get_dart_params
    else:
        raise ValueError(f"Unknown search space: {search_space}")

    gpu_slots = [gpu_id for gpu_id in range(num_gpus) for _ in range(trials_per_gpu)]
    next_slot_idx = 0

    def submit_one(executor: ProcessPoolExecutor):
        nonlocal next_slot_idx
        for attempt in range(10):
            try:
                trial = study.ask()
                break
            except Exception as e:
                print(f"[rank {rank}] study.ask() failed (attempt {attempt}): {e}")
                _sleep_backoff(attempt)
        else:
            raise RuntimeError("study.ask() failed repeatedly")

        params = get_params(trial)

        base_seed = int(trial.number) + 1
        seed = base_seed + rank * 1_000_000
        gpu_id = gpu_slots[next_slot_idx]
        next_slot_idx = (next_slot_idx + 1) % len(gpu_slots)

        fut = executor.submit(
            run_trial_with_params,
            script_path,
            params,
            int(trial.number),
            int(gpu_id),
            int(seed),
            int(total_timesteps),
            env_id,
            int(num_envs),
            int(num_steps),
            fixed_args,
            metric_tag,
            int(last_n),
        )
        return fut, trial

    in_flight: Dict[Any, optuna.Trial] = {}
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=mp.get_context("spawn")) as ex:
        to_launch = min(trials_this_node, total_workers)
        for _ in range(to_launch):
            fut, tr = submit_one(ex)
            in_flight[fut] = tr

        completed = 0
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                tr = in_flight.pop(fut)
                try:
                    value = float(fut.result())
                    state = TrialState.COMPLETE
                except Exception as e:
                    print(f"[rank {rank}] Trial {tr.number} crashed: {e}")
                    value = None
                    state = TrialState.FAIL

                for attempt in range(10):
                    try:
                        if state == TrialState.COMPLETE:
                            study.tell(tr, value)
                        else:
                            study.tell(tr, state=state)
                        break
                    except Exception as e:
                        print(f"[rank {rank}] study.tell() failed (attempt {attempt}): {e}")
                        _sleep_backoff(attempt)

                completed += 1
                if completed >= trials_this_node:
                    continue

                fut2, tr2 = submit_one(ex)
                in_flight[fut2] = tr2

    print(f"[rank {rank}] Coordinator done: completed {completed}/{trials_this_node} trials")


def main():
    parser = argparse.ArgumentParser(description="Generic Optuna tuner for CleanRL scripts")
    parser.add_argument("--script-path", type=str, required=True, help="Path to training script")
    parser.add_argument("--env-id", type=str, required=True, help="Gymnasium env id")
    parser.add_argument("--search-space", type=str, required=True, choices=["ppo", "dart", "ppo_large_critic"],
                        help="Search space to use")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default="postgresql://optuna:optuna_secure_pwd_2026@localhost:5432/optuna_humanoid",
                        help="Optuna storage URL")
    parser.add_argument("--num-gpus", type=int, default=4, help="GPUs per node")
    parser.add_argument("--trials-per-gpu", type=int, default=3, help="Concurrent trials per GPU")
    parser.add_argument("--num-trials", type=int, default=500, help="Total trials across job")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1024)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--metric-tag", type=str, default="charts/episodic_return")
    parser.add_argument("--last-n", type=int, default=50)
    parser.add_argument("--fixed-arg", action="append", default=[],
                        help="Additional fixed arg (key=value). Can be repeated.")

    args = parser.parse_args()

    if args.study_name is None:
        timesteps_m = args.total_timesteps // 1_000_000
        args.study_name = f"{args.search_space}_{args.env_id}_{timesteps_m}M"

    print("=" * 80)
    print("Optuna Generic Tuning")
    print("=" * 80)
    print(f"Study name: {args.study_name}")
    print(f"Script: {args.script_path}")
    print(f"Env: {args.env_id}")
    print(f"Search space: {args.search_space}")
    print(f"Storage: {args.storage}")
    print(f"Total trials: {args.num_trials}")
    print(f"GPUs per node: {args.num_gpus}")
    print(f"Trials per GPU: {args.trials_per_gpu}")
    print(f"Timesteps per trial: {args.total_timesteps:,}")
    print(f"Seeds per trial: {args.num_seeds}")
    print("=" * 80)

    # PostgreSQL connection validation happens implicitly when Optuna connects
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    fixed_args = _parse_kv_list(args.fixed_arg)

    run_node_coordinator(
        study_name=args.study_name,
        storage=args.storage,
        script_path=args.script_path,
        search_space=args.search_space,
        env_id=args.env_id,
        num_gpus=args.num_gpus,
        trials_per_gpu=args.trials_per_gpu,
        total_trials_job=args.num_trials,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        fixed_args=fixed_args,
        metric_tag=args.metric_tag,
        last_n=args.last_n,
    )

    print("=" * 80)
    print(f"All workers completed. Results stored in: {args.storage}")
    print("=" * 80)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()