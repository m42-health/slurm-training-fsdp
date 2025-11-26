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
sbatch --nodes 2 slurm_submit.sh
```

This runs in two phases:
1. **Benchmarks**: Tests storage, CPU, memory, GPU, and network
2. **Training**: ~4B parameter GPT2-style model with FSDP on Salesforce/wikitext (wikitext-103-raw-v1) dataset

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

- **Inter-Node Network**: Latency, bandwidth (ping-pong and all-to-all)
- **Inter-GPU Communication**: All-reduce, all-to-all across all GPUs
- **Storage I/O**: Sequential/parallel read/write, random IOPS (per-node and multi-node aggregate)
- **CPU**: Multi-core stress, matrix ops, compression, sorting
- **Memory**: Bandwidth and allocation performance
- **GPU**: Memory bandwidth, compute (TFLOPS) per device

Results are saved to `logs/benchmark_<jobid>.yaml` (single file from rank 0 with inter-node metrics).

<details>
<summary><b>Example Output (H200, 3 nodes × 8 GPUs)</b></summary>

```yaml
timestamp: 2025-11-21T10:20:36.520830
hostname: worker-3
total_time: 172.49 seconds
results:
  Inter-Node Network:
    Latency (ping-pong): 0.02 ms
    Bandwidth (ping-pong, 24 GPUs): 309.57 GB/s
    All-to-all bandwidth (24 GPUs): 6629.62 GB/s
  GPU 0:
    Memory bandwidth: 3960.35 GB/s
    Compute (FP32 matmul): 6.16 TFLOPS
  GPU 1:
    Memory bandwidth: 3979.42 GB/s
    Compute (FP32 matmul): 51.34 TFLOPS
  GPU 2:
    Memory bandwidth: 3980.88 GB/s
    Compute (FP32 matmul): 51.37 TFLOPS
  GPU 3:
    Memory bandwidth: 3981.07 GB/s
    Compute (FP32 matmul): 51.37 TFLOPS
  GPU 4:
    Memory bandwidth: 3978.05 GB/s
    Compute (FP32 matmul): 51.34 TFLOPS
  GPU 5:
    Memory bandwidth: 3978.61 GB/s
    Compute (FP32 matmul): 51.35 TFLOPS
  GPU 6:
    Memory bandwidth: 3980.17 GB/s
    Compute (FP32 matmul): 51.36 TFLOPS
  GPU 7:
    Memory bandwidth: 3979.32 GB/s
    Compute (FP32 matmul): 51.34 TFLOPS
  Inter-GPU Communication:
    All-reduce (24 GPUs): 200.34 GB/s
    All-to-all (24 GPUs): 56.68 GB/s
  Storage I/O (per-node):
    Sequential Write: 447.99 MB/s
    Sequential Read: 1321.23 MB/s
    Parallel Write (16 workers): 983.30 MB/s
    Parallel Read (16 workers): 1317.59 MB/s
    Random IOPS (4K blocks): 8882.67 ops/s
  Storage I/O (multi-node aggregate):
    Write bandwidth: 9047.42 MB/s
    Read bandwidth: 27009.35 MB/s
  CPU:
    Multi-core stress (all cores): 87.66 K ops/s
    Matrix multiply (NumPy): 94.11 GFLOPS
    Compression (zlib): 46.25 MB/s
    Sorting (NumPy): 75.15 M elements/s
  Memory:
    Bandwidth (copy): 6.38 GB/s
    Allocation/deallocation: 162626.65 ops/s
```

</details>

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
