"""
Analyze Optuna hyperparameter tuning results for PPO vs DART on Humanoid-v4.
Extracts best trials, generates comparison plots, and saves configurations.

Usage:
    python cleanrl_utils/analyze_optuna_results.py --algorithm ppo
    python cleanrl_utils/analyze_optuna_results.py --algorithm dart
    python cleanrl_utils/analyze_optuna_results.py --compare  # Compare both
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from rich.console import Console
from rich.table import Table


console = Console()


def load_study(study_name: str, storage: str) -> optuna.Study:
    """Load Optuna study from storage."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study
    except Exception as e:
        console.print(f"[red]Error loading study '{study_name}': {e}[/red]")
        raise


def get_best_trials(study: optuna.Study, n: int = 10) -> List[optuna.trial.FrozenTrial]:
    """Get top N trials by objective value."""
    all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    all_trials.sort(key=lambda t: t.value, reverse=True)
    return all_trials[:n]


def print_trial_summary(study: optuna.Study, top_n: int = 10):
    """Print summary of study results."""
    console.print(f"\n[bold cyan]Study: {study.study_name}[/bold cyan]")
    console.print(f"Direction: {study.direction.name}")
    console.print(f"Total trials: {len(study.trials)}")
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    console.print(f"  Completed: {len(completed)}")
    console.print(f"  Pruned: {len(pruned)}")
    console.print(f"  Failed: {len(failed)}")
    
    if completed:
        values = [t.value for t in completed]
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  Best: {max(values):.2f}")
        console.print(f"  Mean: {np.mean(values):.2f}")
        console.print(f"  Std: {np.std(values):.2f}")
        console.print(f"  Median: {np.median(values):.2f}")
        
        # Top trials table
        best_trials = get_best_trials(study, n=top_n)
        
        table = Table(title=f"Top {len(best_trials)} Trials")
        table.add_column("Rank", style="cyan")
        table.add_column("Trial", style="magenta")
        table.add_column("Value", style="green")
        table.add_column("Params", style="yellow")
        
        for rank, trial in enumerate(best_trials, 1):
            params_str = ", ".join([f"{k}={v:.4g}" for k, v in trial.params.items()])
            table.add_row(
                str(rank),
                str(trial.number),
                f"{trial.value:.2f}",
                params_str
            )
        
        console.print(table)


def save_best_configs(study: optuna.Study, output_dir: Path, top_n: int = 3):
    """Save top N configurations to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_trials = get_best_trials(study, n=top_n)
    
    for rank, trial in enumerate(best_trials, 1):
        config = {
            "study_name": study.study_name,
            "trial_number": trial.number,
            "objective_value": trial.value,
            "params": trial.params,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        }
        
        output_file = output_dir / f"{study.study_name}_rank{rank}_trial{trial.number}.json"
        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)
        
        console.print(f"Saved config: {output_file}")


def plot_optimization_history(study: optuna.Study, output_dir: Path):
    """Plot optimization history."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed:
        console.print("[yellow]No completed trials to plot[/yellow]")
        return
    
    trial_numbers = [t.number for t in completed]
    values = [t.value for t in completed]
    
    # Compute running best
    running_best = []
    best_so_far = float('-inf')
    for v in values:
        best_so_far = max(best_so_far, v)
        running_best.append(best_so_far)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(trial_numbers, values, alpha=0.5, s=20, label='Trial value')
    ax.plot(trial_numbers, running_best, 'r-', linewidth=2, label='Best value')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Objective Value (Episodic Return)')
    ax.set_title(f'Optimization History: {study.study_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_file = output_dir / f"{study.study_name}_history.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"Saved plot: {output_file}")


def plot_param_importances(study: optuna.Study, output_dir: Path):
    """Plot parameter importances."""
    try:
        importances = optuna.importance.get_param_importances(study)
        
        if not importances:
            console.print("[yellow]No parameter importances to plot[/yellow]")
            return
        
        params = list(importances.keys())
        values = list(importances.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(params, values)
        ax.set_xlabel('Importance')
        ax.set_title(f'Parameter Importances: {study.study_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        output_file = output_dir / f"{study.study_name}_importances.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        console.print(f"Saved plot: {output_file}")
    except Exception as e:
        console.print(f"[yellow]Could not compute parameter importances: {e}[/yellow]")


def compare_algorithms(
    ppo_study: optuna.Study,
    dart_study: optuna.Study,
    output_dir: Path
):
    """Compare PPO and DART results."""
    ppo_completed = [t for t in ppo_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    dart_completed = [t for t in dart_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not ppo_completed or not dart_completed:
        console.print("[yellow]Need completed trials for both algorithms to compare[/yellow]")
        return
    
    # Statistical comparison
    ppo_values = [t.value for t in ppo_completed]
    dart_values = [t.value for t in dart_completed]
    
    console.print("\n[bold cyan]PPO vs DART Comparison[/bold cyan]")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("PPO", style="green")
    table.add_column("DART", style="magenta")
    
    table.add_row("Trials", str(len(ppo_values)), str(len(dart_values)))
    table.add_row("Best", f"{max(ppo_values):.2f}", f"{max(dart_values):.2f}")
    table.add_row("Mean", f"{np.mean(ppo_values):.2f}", f"{np.mean(dart_values):.2f}")
    table.add_row("Median", f"{np.median(ppo_values):.2f}", f"{np.median(dart_values):.2f}")
    table.add_row("Std", f"{np.std(ppo_values):.2f}", f"{np.std(dart_values):.2f}")
    
    console.print(table)
    
    # Box plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([ppo_values, dart_values], labels=['PPO', 'DART'])
    ax.set_ylabel('Episodic Return')
    ax.set_title('PPO vs DART: Distribution of Trial Returns')
    ax.grid(True, alpha=0.3, axis='y')
    
    output_file = output_dir / "ppo_vs_dart_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"Saved comparison plot: {output_file}")
    
    # Best trial comparison
    ppo_best = get_best_trials(ppo_study, n=1)[0]
    dart_best = get_best_trials(dart_study, n=1)[0]
    
    console.print("\n[bold]Best Configurations:[/bold]")
    console.print(f"\n[green]PPO (Trial {ppo_best.number}, Return: {ppo_best.value:.2f})[/green]")
    for k, v in ppo_best.params.items():
        console.print(f"  {k}: {v}")
    
    console.print(f"\n[magenta]DART (Trial {dart_best.number}, Return: {dart_best.value:.2f})[/magenta]")
    for k, v in dart_best.params.items():
        console.print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna tuning results")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "dart"],
                        help="Algorithm to analyze (ppo or dart)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare PPO and DART results")
    parser.add_argument("--storage", type=str, default="sqlite:////scratch/shahradm/optuna_humanoid.db",
                        help="Optuna storage URL")
    parser.add_argument("--output-dir", type=str, default="/scratch/shahradm/optuna_results",
                        help="Output directory for plots and configs")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top trials to display/save")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # Compare both algorithms
        ppo_study = load_study("ppo_humanoid_50M", args.storage)
        dart_study = load_study("dart_humanoid_50M", args.storage)
        
        print_trial_summary(ppo_study, top_n=args.top_n)
        print_trial_summary(dart_study, top_n=args.top_n)
        
        compare_algorithms(ppo_study, dart_study, output_dir)
        
        # Save best configs for both
        save_best_configs(ppo_study, output_dir / "ppo", top_n=3)
        save_best_configs(dart_study, output_dir / "dart", top_n=3)
        
    elif args.algorithm:
        # Analyze single algorithm
        study_name = f"{args.algorithm}_humanoid_50M"
        study = load_study(study_name, args.storage)
        
        print_trial_summary(study, top_n=args.top_n)
        
        # Generate plots
        plot_optimization_history(study, output_dir)
        plot_param_importances(study, output_dir)
        
        # Save best configs
        save_best_configs(study, output_dir / args.algorithm, top_n=3)
    
    else:
        console.print("[red]Error: Must specify --algorithm or --compare[/red]")
        return
    
    console.print("\n[bold green]Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
