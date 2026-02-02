#!/bin/bash
#SBATCH --job-name=final_comparison
#SBATCH --partition=production-cluster
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --account=aip-rrabba
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shahradm/slurm_logs/final_comparison_%A_%a.out
#SBATCH --error=/scratch/shahradm/slurm_logs/final_comparison_%A_%a.err

# ================================================================================
# Final validation: Run top-3 configs from PPO and DART with 5 seeds each
# 
# Usage:
#   # First, extract best configs from Optuna results
#   python cleanrl_utils/analyze_optuna_results.py --compare
#   
#   # Then run this script
#   sbatch benchmark/final_comparison.sh
# ================================================================================

# Configuration
RESULTS_DIR="/scratch/shahradm/optuna_results"
FINAL_RUNS_DIR="/scratch/shahradm/final_comparison_runs"
TOTAL_TIMESTEPS=50000000
NUM_SEEDS=5

# Create directories
mkdir -p $FINAL_RUNS_DIR
mkdir -p /scratch/shahradm/slurm_logs

echo "========================================================================"
echo "Final Comparison: Top-3 PPO vs Top-3 DART configs"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Total runs: 30 (3 configs × 2 algorithms × 5 seeds)"
echo "========================================================================"

# Load environment (uv)
source .env

# Function to run a single experiment
run_experiment() {
    local algorithm=$1
    local config_file=$2
    local seed=$3
    local gpu_id=$4
    
    local config_name=$(basename $config_file .json)
    local exp_name="${algorithm}_${config_name}_seed${seed}"
    
    echo "[GPU $gpu_id] Running: $exp_name"
    
    # Extract parameters from JSON and build command
    if [ "$algorithm" == "ppo" ]; then
        script="cleanrl/ppo_humanoid_sparse.py"
    else
        script="cleanrl/dart_humanoid_sparse_opt.py"
    fi
    
    # Parse JSON and construct arguments (requires jq)
    local lr=$(jq -r '.params["learning-rate"]' $config_file)
    local gae_lambda=$(jq -r '.params["gae-lambda"]' $config_file)
    local ent_coef=$(jq -r '.params["ent-coef"] // 0.01' $config_file)
    local clip_coef=$(jq -r '.params["clip-coef"] // 0.2' $config_file)
    local num_mb=$(jq -r '.params["num-minibatches"]' $config_file)
    local update_epochs=$(jq -r '.params["update-epochs"]' $config_file)
    
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python $script \
        --exp-name $exp_name \
        --seed $seed \
        --total-timesteps $TOTAL_TIMESTEPS \
        --learning-rate $lr \
        --gae-lambda $gae_lambda \
        --ent-coef $ent_coef \
        --clip-coef $clip_coef \
        --num-minibatches $num_mb \
        --update-epochs $update_epochs \
        --num-envs 64 \
        --num-steps 1024 \
        --vectorization async \
        --torch-compile \
        --track \
        --wandb-project-name humanoid_ppo_vs_dart_final \
        --save-model \
        --capture-video"
    
    # Add DART-specific params if needed
    if [ "$algorithm" == "dart" ]; then
        local dart_lr_scale=$(jq -r '.params["dart-lr-scale"]' $config_file)
        local dart_lambda=$(jq -r '.params["dart-lambda-res"]' $config_file)
        local dart_warmup=$(jq -r '.params["dart-warmup-frac"]' $config_file)
        cmd="$cmd --dart-lr-scale $dart_lr_scale --dart-lambda-res $dart_lambda --dart-warmup-frac $dart_warmup"
    fi
    
    eval $cmd
    
    echo "[GPU $gpu_id] Completed: $exp_name"
}

export -f run_experiment
export TOTAL_TIMESTEPS
export FINAL_RUNS_DIR

# Build list of all runs
runs_file="/tmp/final_comparison_runs_$$.txt"
touch $runs_file

# Add PPO runs
for rank in 1 2 3; do
    config_file="$RESULTS_DIR/ppo/ppo_humanoid_50M_rank${rank}_trial*.json"
    if ls $config_file 1> /dev/null 2>&1; then
        config_file=$(ls $config_file | head -1)
        for seed in $(seq 1 $NUM_SEEDS); do
            echo "ppo $config_file $seed" >> $runs_file
        done
    fi
done

# Add DART runs
for rank in 1 2 3; do
    config_file="$RESULTS_DIR/dart/dart_humanoid_50M_rank${rank}_trial*.json"
    if ls $config_file 1> /dev/null 2>&1; then
        config_file=$(ls $config_file | head -1)
        for seed in $(seq 1 $NUM_SEEDS); do
            echo "dart $config_file $seed" >> $runs_file
        done
    fi
done

# Check if we have runs
total_runs=$(wc -l < $runs_file)
echo "Found $total_runs runs to execute"

if [ $total_runs -eq 0 ]; then
    echo "ERROR: No configuration files found in $RESULTS_DIR"
    echo "Make sure to run analyze_optuna_results.py first!"
    exit 1
fi

# Run all experiments in parallel using GNU parallel or simple loop with background jobs
# Using 4 GPUs in parallel
gpu=0
while IFS=' ' read -r algorithm config_file seed; do
    run_experiment "$algorithm" "$config_file" "$seed" "$gpu" &
    
    # Cycle through GPUs
    gpu=$(( (gpu + 1) % 4 ))
    
    # Wait if we've filled all GPUs
    if [ $gpu -eq 0 ]; then
        wait
    fi
done < $runs_file

# Wait for remaining jobs
wait

# Clean up
rm -f $runs_file

echo "========================================================================"
echo "All final comparison runs completed!"
echo "Results saved to runs/ directory"
echo "========================================================================"
