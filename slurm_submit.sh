#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=1          # One task per node (torchrun handles the rest)
#SBATCH --gpus-per-node=8            # Request 8 GPUs per node
#SBATCH --cpus-per-task=64           # CPU cores per node
# #SBATCH --time=02:00:00              # Maximum runtime (2 hours)
# #SBATCH --partition=gpu            # Partition name (adjust for your cluster)
#SBATCH --output=logs/train_%j.out   # Standard output log
#SBATCH --error=logs/train_%j.err    # Standard error log
#SBATCH --exclusive                  # Exclusive node access

# Create logs directory
mkdir -p logs

echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"

# Activate uv environment
source .venv/bin/activate

export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=8388608
export NCCL_IB_CUDA_SUPPORT=1
export WANDB__SERVICE_WAIT=300
export PYTHONFAULTHANDLER=1

# Set up master node
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Set HuggingFace cache directories to shared storage
# HF_HOME sets the base directory for all HuggingFace caches
export HF_HOME=/m42pfsdata/hf_cache
# Explicit cache paths for different components
export HF_DATASETS_CACHE=/m42pfsdata/hf_cache/datasets
export HF_HUB_CACHE=/m42pfsdata/hf_cache/hub

# Create cache directories if they don't exist
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_HUB_CACHE

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "HuggingFace cache directories:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"

# ============================================================================
# BENCHMARK PHASE - Test all system resources before training
# ============================================================================
echo ""
echo "=============================================================================="
echo "PHASE 1: Running comprehensive benchmarks"
echo "=============================================================================="

# Run benchmarks with distributed setup
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    benchmark.py

echo ""
echo "Benchmarks completed at: $(date)"
echo ""

# ============================================================================
# TRAINING PHASE
# ============================================================================
echo "=============================================================================="
echo "PHASE 2: Starting training"
echo "=============================================================================="

# Run training with torchrun via srun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py

echo ""
echo "Job finished at: $(date)"

