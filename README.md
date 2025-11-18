# FSDP Training with PyTorch and Transformers

Simple, readable PyTorch FSDP training script for multi-node SLURM clusters with ~4B parameter GPT2-style model.

## Quick Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Submit Training Job

```bash
# Edit slurm_submit.sh to match your cluster (nodes, GPUs, partition, etc.)
# Then submit:
sbatch slurm_submit.sh
```

## Monitor Job

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/train_JOBID.out
```

## Configuration

### SLURM Settings (slurm_submit.sh)

Default configuration uses 4 nodes with 8 GPUs each (32 GPUs total):

```bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
```

Adjust based on your cluster. Common configurations:
- Single node: `--nodes=1 --gpus-per-node=8`
- Two nodes: `--nodes=2 --gpus-per-node=8`

### Training Settings (train.py)

Key parameters you can adjust:

```python
# Model size (~4B params)
n_embd=2560        # Embedding dimension
n_layer=32         # Number of layers
n_head=32          # Attention heads

# Training
batch_size=2       # Per-device batch size
seq_length=512     # Sequence length
num_epochs=3       # Training epochs
lr=1e-4           # Learning rate
```

## How It Works

- **Model**: ~4B parameter GPT2-style transformer
- **Data**: Randomly generated dummy data (for testing)
- **Distributed**: FSDP with full sharding across all GPUs
- **Launcher**: `srun torchrun` for multi-node coordination
- **Checkpointing**: Saves to `./checkpoints/final_checkpoint.pt`

## File Structure

```
├── train.py              # Main training script
├── requirements.txt      # Dependencies
├── slurm_submit.sh      # SLURM job script
└── README.md            # This file
```

## Troubleshooting

### Out of Memory

Reduce per device batch size or sequence length in `train.py`:
```python
batch_size=1       # Instead of 16
seq_length=256     # Instead of 512
```

### Job Stuck in Queue

Check available resources:
```bash
sinfo              # Check cluster status
squeue -u $USER    # Check your jobs
```

## Customization

### Use Real Data

Replace `DummyTextDataset` in `train.py`:

```python
from datasets import load_dataset
dataset = load_dataset("your_dataset")
```

### Use Different Model

Replace model creation in `train.py`:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

## Requirements

- Python 3.11+
- CUDA-capable GPUs
- SLURM cluster
- uv package manager

## License

Demonstration code - free to use and modify.
