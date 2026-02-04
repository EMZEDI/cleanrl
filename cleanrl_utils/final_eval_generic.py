"""
Generic final-evaluation runner for CleanRL-style scripts.

Loads top-N config JSONs (saved by analyze_optuna_results.py) and runs each with
N seeds, distributing runs across available GPUs on a single node.
"""

import argparse
import inspect
import json
import os
import runpy
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch


def _parse_kv_list(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --fixed-arg '{item}', expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_configs(config_dir: Path, pattern: str, top_n: int) -> List[Path]:
    configs = sorted(config_dir.glob(pattern))
    if not configs:
        return []

    def _score(p: Path) -> float:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            return float(data.get("objective_value", float("-inf")))
        except Exception:
            return float("-inf")

    configs.sort(key=_score, reverse=True)
    return configs[:top_n]


def _call_make_env(make_env, env_id: str, run_name: str, gamma: Optional[float]):
    sig = inspect.signature(make_env)
    kwargs = {}
    if "env_id" in sig.parameters:
        kwargs["env_id"] = env_id
    if "idx" in sig.parameters:
        kwargs["idx"] = 0
    if "capture_video" in sig.parameters:
        kwargs["capture_video"] = False
    if "run_name" in sig.parameters:
        kwargs["run_name"] = run_name
    if "gamma" in sig.parameters and gamma is not None:
        kwargs["gamma"] = gamma
    return make_env(**kwargs)


def _evaluate_agent(module_globals: Dict[str, Any], env_id: str, eval_episodes: int, seed: int) -> Dict[str, Any]:
    if eval_episodes <= 0:
        return {}

    if "Agent" not in module_globals or "make_env" not in module_globals:
        raise RuntimeError("Script must define Agent and make_env for evaluation")

    Agent = module_globals["Agent"]
    make_env = module_globals["make_env"]
    args = module_globals.get("args", None)
    gamma = getattr(args, "gamma", None) if args is not None else None

    env = gym.vector.SyncVectorEnv([_call_make_env(make_env, env_id, "eval", gamma)])

    # Use the trained agent if available; otherwise instantiate.
    agent = module_globals.get("agent", None)
    if agent is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = Agent(env).to(device)
        state_dict = module_globals.get("agent_state_dict", None)
        if state_dict is not None:
            agent.load_state_dict(state_dict)
    else:
        device = next(agent.parameters()).device

    agent.eval()

    obs, _ = env.reset(seed=seed)
    returns: List[float] = []

    while len(returns) < eval_episodes:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            if hasattr(agent, "actor_mean"):
                action = agent.actor_mean(obs_t)
            else:
                action, *_ = agent.get_action_and_value(obs_t)
        obs, _, terminations, truncations, infos = env.step(action.detach().cpu().numpy())

        if isinstance(infos, dict) and "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    returns.append(float(info["episode"]["r"]))

    env.close()

    return {
        "eval_episodes": eval_episodes,
        "eval_mean_return": float(np.mean(returns)) if returns else float("-inf"),
        "eval_returns": returns,
    }


def _run_one(
    script_path: str,
    params: Dict[str, Any],
    seed: int,
    gpu_id: int,
    env_id: str,
    total_timesteps: int,
    num_envs: int,
    num_steps: int,
    fixed_args: Dict[str, str],
    exp_name: str,
    eval_episodes: int,
    eval_results_dir: str,
):
    fixed_params = {
        "env-id": env_id,
        "total-timesteps": total_timesteps,
        "num-envs": num_envs,
        "num-steps": num_steps,
        "seed": seed,
        "exp-name": exp_name,
    }
    all_params: Dict[str, Any] = {**fixed_params, **fixed_args, **params}

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.argv = [script_path] + [f"--{key}={value}" for key, value in all_params.items()]
    module_globals = runpy.run_path(path_name=script_path, run_name="__main__")

    eval_summary = _evaluate_agent(module_globals, env_id, eval_episodes, seed)
    if eval_summary:
        Path(eval_results_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(eval_results_dir) / f"{exp_name}_eval.json"
        with open(out_path, "w") as f:
            json.dump(eval_summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generic final evaluation runner")
    parser.add_argument("--script-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--config-pattern", type=str, default="*_rank*_trial*.json")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--parallel-per-gpu", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes after training")
    parser.add_argument("--eval-results-dir", type=str, default="/scratch/shahradm/final_eval_results",
                        help="Directory to store eval JSON summaries")
    parser.add_argument("--fixed-arg", action="append", default=[],
                        help="Additional fixed arg (key=value). Can be repeated.")
    parser.add_argument("--exp-prefix", type=str, default="final_eval",
                        help="Prefix for exp-name")

    args = parser.parse_args()
    fixed_args = _parse_kv_list(args.fixed_arg)

    config_dir = Path(args.config_dir)
    configs = _load_configs(config_dir, args.config_pattern, args.top_n)
    if not configs:
        raise SystemExit(f"No config files found in {config_dir} matching {args.config_pattern}")

    total_workers = args.num_gpus * args.parallel_per_gpu
    gpu_slots = [gpu_id for gpu_id in range(args.num_gpus) for _ in range(args.parallel_per_gpu)]
    slot_idx = 0

    jobs: List[Tuple[Path, int, int]] = []
    for cfg in configs:
        for seed in range(1, args.num_seeds + 1):
            gpu_id = gpu_slots[slot_idx]
            slot_idx = (slot_idx + 1) % len(gpu_slots)
            jobs.append((cfg, seed, gpu_id))

    print(f"Running {len(jobs)} runs across {total_workers} workers")

    with ProcessPoolExecutor(max_workers=total_workers) as ex:
        futures = []
        for cfg_path, seed, gpu_id in jobs:
            with open(cfg_path, "r") as f:
                data = json.load(f)
            params = data.get("params", {})
            exp_name = f"{args.exp_prefix}_{cfg_path.stem}_seed{seed}"
            futures.append(
                ex.submit(
                    _run_one,
                    args.script_path,
                    params,
                    seed,
                    gpu_id,
                    args.env_id,
                    args.total_timesteps,
                    args.num_envs,
                    args.num_steps,
                    fixed_args,
                    exp_name,
                    args.eval_episodes,
                    args.eval_results_dir,
                )
            )

        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"Run failed: {exc}")


if __name__ == "__main__":
    main()