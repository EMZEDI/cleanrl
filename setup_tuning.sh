#!/bin/bash
# Setup script for hyperparameter tuning infrastructure

set -e

echo "========================================================================"
echo "Setting up hyperparameter tuning environment"
echo "========================================================================"

# Configuration
SCRATCH_DIR="/scratch/shahradm"
DB_PATH="$SCRATCH_DIR/optuna_humanoid.db"
RESULTS_DIR="$SCRATCH_DIR/optuna_results"
LOGS_DIR="$SCRATCH_DIR/slurm_logs"

# Load environment (uv)
if [ -f ".env" ]; then
    echo "Loading environment from .env..."
    source .env
    echo "✓ Environment loaded"
else
    echo "WARNING: .env file not found in current directory"
fi

# 1. Create directories
echo "Creating directories..."
mkdir -p "$RESULTS_DIR"/{ppo,dart}
mkdir -p "$LOGS_DIR"

echo "✓ Directories created"

# 2. Check Python/uv environment
echo "Checking Python environment..."

if ! command -v python &> /dev/null; then
    echo "ERROR: python not found. Make sure .env loads uv correctly."
    exit 1
fi

echo "Python: $(which python)"
echo "✓ Python environment ready"

# 3. Test SQLite write access
echo "Testing database write access..."
touch "$DB_PATH"
if [ ! -w "$DB_PATH" ]; then
    echo "ERROR: Cannot write to $DB_PATH"
    exit 1
fi
rm -f "$DB_PATH"
echo "✓ Database path writable"

# 4. Test GPU access (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Testing GPU access..."
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "✓ Found $GPU_COUNT GPUs"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv
else
    echo "⚠ nvidia-smi not found (OK if running on head node)"
fi

# 5. Make scripts executable
echo "Making scripts executable..."
chmod +x benchmark/tune_humanoid_820gpu.sh
chmod +x benchmark/final_comparison.sh
echo "✓ Scripts are executable"

# 6. Verify scripts exist
echo "Verifying tuning scripts..."
REQUIRED_SCRIPTS=(
    "cleanrl_utils/tune_ppo_dart_humanoid.py"
    "cleanrl_utils/analyze_optuna_results.py"
    "benchmark/tune_humanoid_820gpu.sh"
    "benchmark/final_comparison.sh"
    "cleanrl/ppo_humanoid_sparse.py"
    "cleanrl/dart_humanoid_sparse_opt.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Missing script: $script"
        exit 1
    fi
done

echo "✓ All scripts present"

# 7. Test dry run
echo ""
echo "Running test trial (dry run)..."

python cleanrl_utils/tune_ppo_dart_humanoid.py \
    --algorithm ppo \
    --num-trials 1 \
    --num-gpus 1 \
    --total-timesteps 1000 \
    --single-gpu \
    2>&1 | head -20

echo ""
echo "========================================================================"
echo "Setup complete! ✓"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Launch hyperparameter tuning:"
echo "   sbatch benchmark/tune_humanoid_820gpu.sh ppo"
echo "   sbatch benchmark/tune_humanoid_820gpu.sh dart"
echo ""
echo "2. Monitor progress:"
echo "   squeue -u \$USER"
echo "   tail -f $LOGS_DIR/tune_humanoid_*.out"
echo ""
echo "3. After completion, analyze results:"
echo "   python cleanrl_utils/analyze_optuna_results.py --compare"
echo ""
echo "4. Run final validation:"
echo "   sbatch benchmark/final_comparison.sh"
echo ""
echo "See TUNING_GUIDE.md for full documentation"
echo "========================================================================"
