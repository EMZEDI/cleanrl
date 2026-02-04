#!/bin/bash

# Submit tuning for PPO, DART, and PPO-large-critic, then run analysis and final eval (20 seeds).
# Usage:
#   bash benchmark/pipeline_submit.sh <env_id> <ppo_script> <dart_script> <ppo_large_script> [timesteps] [num_trials]

set -euo pipefail

ENV_ID=${1:?"env_id required"}
PPO_SCRIPT=${2:?"ppo script path required"}
DART_SCRIPT=${3:?"dart script path required"}
PPO_LARGE_SCRIPT=${4:?"ppo_large_critic script path required"}
TOTAL_TIMESTEPS=${5:-50000000}
NUM_TRIALS=${6:-500}

TIMESTEPS_M=$((TOTAL_TIMESTEPS/1000000))
STUDY_PPO="ppo_${ENV_ID}_${TIMESTEPS_M}M"
STUDY_DART="dart_${ENV_ID}_${TIMESTEPS_M}M"
STUDY_PPO_LARGE="ppo_large_critic_${ENV_ID}_${TIMESTEPS_M}M"

echo "Submitting tuning jobs..."

JOB_PPO=$(sbatch benchmark/tune_generic.sh "$ENV_ID" ppo "$PPO_SCRIPT" "$TOTAL_TIMESTEPS" "$NUM_TRIALS" | awk '{print $4}')
JOB_DART=$(sbatch benchmark/tune_generic.sh "$ENV_ID" dart "$DART_SCRIPT" "$TOTAL_TIMESTEPS" "$NUM_TRIALS" | awk '{print $4}')
JOB_PPO_LARGE=$(sbatch benchmark/tune_generic.sh "$ENV_ID" ppo_large_critic "$PPO_LARGE_SCRIPT" "$TOTAL_TIMESTEPS" "$NUM_TRIALS" | awk '{print $4}')

echo "PPO job: $JOB_PPO"
echo "DART job: $JOB_DART"
echo "PPO Large Critic job: $JOB_PPO_LARGE"

echo "Submitting analysis job (dependent)..."
ANALYSIS_CMD="source .env; python cleanrl_utils/analyze_optuna_results.py \
  --study-names ${STUDY_PPO},${STUDY_DART},${STUDY_PPO_LARGE} \
  --labels PPO,DART,PPO_LARGE \
  --top-n 10"

JOB_ANALYSIS=$(sbatch \
  --job-name=analyze_generic \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --account=aip-rrabba \
  --output=/scratch/shahradm/slurm_logs/analyze_generic_%j.out \
  --error=/scratch/shahradm/slurm_logs/analyze_generic_%j.err \
  --dependency=afterok:${JOB_PPO}:${JOB_DART}:${JOB_PPO_LARGE} \
  --wrap "$ANALYSIS_CMD" | awk '{print $4}')

echo "Analysis job: $JOB_ANALYSIS"

echo "Submitting final eval jobs (dependent on analysis)..."
NUM_SEEDS_FINAL=${NUM_SEEDS_FINAL:-20}

JOB_EVAL_PPO=$(sbatch --dependency=afterok:${JOB_ANALYSIS} \
  benchmark/final_eval_generic.sh "$ENV_ID" "$PPO_SCRIPT" "/scratch/shahradm/optuna_results/${STUDY_PPO}" "$TOTAL_TIMESTEPS" "$NUM_SEEDS_FINAL" | awk '{print $4}')

JOB_EVAL_DART=$(sbatch --dependency=afterok:${JOB_ANALYSIS} \
  benchmark/final_eval_generic.sh "$ENV_ID" "$DART_SCRIPT" "/scratch/shahradm/optuna_results/${STUDY_DART}" "$TOTAL_TIMESTEPS" "$NUM_SEEDS_FINAL" | awk '{print $4}')

JOB_EVAL_PPO_LARGE=$(sbatch --dependency=afterok:${JOB_ANALYSIS} \
  benchmark/final_eval_generic.sh "$ENV_ID" "$PPO_LARGE_SCRIPT" "/scratch/shahradm/optuna_results/${STUDY_PPO_LARGE}" "$TOTAL_TIMESTEPS" "$NUM_SEEDS_FINAL" | awk '{print $4}')

echo "Final eval jobs:"
echo "  PPO: $JOB_EVAL_PPO"
echo "  DART: $JOB_EVAL_DART"
echo "  PPO Large Critic: $JOB_EVAL_PPO_LARGE"

echo "Pipeline submitted."