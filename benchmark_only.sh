#!/bin/bash
#SBATCH --job-name=cluster-benchmark
#SBATCH --nodes=4                    # Number of nodes to test
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --gpus-per-node=8            # GPUs to test per node
#SBATCH --cpus-per-task=64           # CPU cores per node
#SBATCH --time=00:30:00              # 30 minutes should be enough
# #SBATCH --partition=gpu            # Partition name (adjust for your cluster)
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --exclusive                  # Exclusive node access

# Create logs directory
mkdir -p logs

echo "=============================================================================="
echo "                    SLURM CLUSTER BENCHMARK SUITE"
echo "=============================================================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Started at:   $(date)"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Node count:   $SLURM_JOB_NUM_NODES"
echo "GPUs/node:    $SLURM_GPUS_PER_NODE"
echo "CPUs/node:    $SLURM_CPUS_ON_NODE"
echo "=============================================================================="
echo ""

# Activate uv environment
source .venv/bin/activate

# Set up environment
export NCCL_DEBUG=INFO
export NCCL_BUFFSIZE=8388608
export NCCL_IB_CUDA_SUPPORT=1
export PYTHONFAULTHANDLER=1

# Set up master node for distributed benchmarks
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master node:  $MASTER_ADDR:$MASTER_PORT"
echo ""

# Run benchmarks with distributed setup
echo "Running benchmarks..."
echo ""

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    benchmark.py

echo ""
echo "=============================================================================="
echo "Benchmark completed at: $(date)"
echo "Results saved to: logs/benchmark_${SLURM_JOB_ID}.json"
echo "=============================================================================="

