# FSDP Training with Benchmarks

PyTorch FSDP training script for multi-node SLURM clusters with comprehensive benchmark suite to test cluster resources before training.

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Full Run (Benchmarks + Training)

```bash
sbatch slurm_submit.sh
```

This runs in two phases:
1. **Benchmarks**: Tests storage, CPU, memory, GPU, and network
2. **Training**: ~4B parameter GPT2-style model with FSDP

### Benchmarks Only

```bash
sbatch benchmark_only.sh
```

### Monitor

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/train_*.out

# View benchmark results (per rank in multi-node)
cat logs/benchmark_*_rank*.yaml
```

## Configuration

### SLURM Settings

Edit `slurm_submit.sh` or `benchmark_only.sh`:

```bash
#SBATCH --nodes=4              # Number of nodes
#SBATCH --gpus-per-node=8      # GPUs per node
#SBATCH --cpus-per-task=64     # CPUs per node
#SBATCH --time=02:00:00        # Max runtime
#SBATCH --partition=gpu        # Partition (uncomment and set)
```

### Training Settings

Edit `train.py`:

```python
n_embd=2560        # Model dimension
n_layer=32         # Number of layers
n_head=32          # Attention heads
batch_size=2       # Per-device batch size
seq_length=512     # Sequence length
num_epochs=3       # Training epochs
```

## Benchmarks

The benchmark suite tests:

- **Storage I/O**: Sequential/parallel read/write, random IOPS
- **CPU**: Multi-core stress, matrix ops, compression, sorting
- **Memory**: Bandwidth and allocation performance
- **GPU**: Memory bandwidth, compute (TFLOPS), all-reduce (multi-node)
- **Network**: Node-to-node latency and bandwidth (multi-node)

Results are saved to `logs/benchmark_<jobid>_rank<N>.yaml` (one file per rank/node).

### Expected Performance

- **Storage**: NVMe SSD (3-7 GB/s), Network FS (1-10 GB/s)
- **Memory**: DDR4 (20-40 GB/s), DDR5 (40-80 GB/s)
- **GPU (A100)**: ~1500-2000 GB/s memory, ~150-312 TFLOPS
- **Network**: InfiniBand (12-25 GB/s, <2μs latency)

## File Structure

```
.
├── train.py              # Training script
├── benchmark.py          # Benchmark suite
├── slurm_submit.sh      # Benchmarks + training job
├── benchmark_only.sh    # Benchmarks only job
├── requirements.txt      # Dependencies
├── logs/                 # Output logs (auto-generated)
└── checkpoints/          # Model checkpoints (auto-generated)
```

## Troubleshooting

### Out of Memory

Reduce batch size or sequence length in `train.py`:

```python
batch_size=1
seq_length=256
```

### Job Issues

```bash
sinfo                      # Check cluster status
squeue -u $USER            # Check your jobs
scancel <jobid>            # Cancel a job
tail -f logs/train_*.err   # Check errors
```

## Requirements

- Python 3.11+
- CUDA-capable GPUs
- SLURM cluster
- uv package manager

For more details on benchmarks, see `docs/BENCHMARK_FEATURES.md`.
