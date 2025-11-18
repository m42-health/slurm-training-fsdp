#!/usr/bin/env python3
"""
Simple FSDP Training Script with Transformers
Uses dummy data for demonstration purposes.
"""

import os
import time
from datetime import datetime
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import functools
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


class DummyTextDataset(Dataset):
    """Generate random token sequences for training."""

    def __init__(self, num_samples=10000, seq_length=512, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random token ids
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),  # For language modeling, labels = inputs shifted
        }


def setup_distributed():
    """Initialize distributed training environment."""
    # Try standard PyTorch distributed variables first, then fall back to SLURM
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    # Set up the process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set device
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def get_timestamp():
    """Get formatted timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_model():
    """Create a ~4B parameter GPT2-like model."""
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=2560,  # Embedding dimension
        n_layer=32,  # Number of transformer layers
        n_head=32,  # Number of attention heads
        n_inner=10240,  # FFN inner dimension (4x n_embd)
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        loss_type="ForCausalLMLoss",  # Explicitly set to suppress warning
    )

    model = GPT2LMHeadModel(config)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())

    return model, num_params


def train(rank, world_size, local_rank):
    """Main training loop."""

    print(f"[Rank {rank}] Starting training on GPU {local_rank}")

    # Create model
    model, num_params = create_model()
    print(
        f"[Rank {rank}] Model created with {num_params:,} parameters ({num_params/1e9:.2f}B)"
    )

    # Move model to GPU
    model = model.to(local_rank)

    # Wrap model with FSDP
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Fully shard parameters
        device_id=local_rank,
        limit_all_gathers=True,
    )

    print(f"[Rank {rank}] Model wrapped with FSDP")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Create dataset and dataloader
    dataset = DummyTextDataset(num_samples=10000, seq_length=512)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Per-device batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[Rank {rank}] DataLoader created")

    # Training loop
    num_epochs = 3
    model.train()

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently each epoch

        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        # For tracking step times over logging interval
        interval_start_time = time.time()
        interval_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to GPU
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            interval_steps += 1

            # Log every 10 steps
            if batch_idx % 10 == 0 and rank == 0:
                interval_time = time.time() - interval_start_time
                avg_step_time = (
                    interval_time / interval_steps if interval_steps > 0 else 0
                )
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] [Step {batch_idx}] "
                    f"Loss: {loss.item():.4f} | Avg step time: {avg_step_time:.3f}s"
                )
                # Reset interval tracking
                interval_start_time = time.time()
                interval_steps = 0

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start_time

        if rank == 0:
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Average Loss: {avg_loss:.4f} | Epoch time: {epoch_time:.2f}s"
            )

    # Save checkpoint
    if rank == 0:
        print("Saving model...")

    checkpoint_dir = "./checkpoints"
    save_start_time = time.time()

    # Load state dict to CPU before saving
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        fullstate_save_policy,
    ):
        cpu_state = model.state_dict()

    if rank == 0:
        model.module.save_pretrained(
            checkpoint_dir,
            safe_serialization=True,
            state_dict=cpu_state,
        )
        save_time = time.time() - save_start_time
        print(f"Model saved to {checkpoint_dir} (took {save_time:.2f}s)")

    print(f"[Rank {rank}] Training completed!")


def main():
    """Main entry point."""

    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 50)
        print(f"FSDP Training with Transformers")
        print("=" * 50)
        print(f"World size: {world_size}")
        print(f"Using {torch.cuda.device_count()} GPUs per node")
        print("=" * 50)

    try:
        # Run training
        train(rank, world_size, local_rank)
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    main()
