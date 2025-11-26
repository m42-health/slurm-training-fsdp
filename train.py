#!/usr/bin/env python3
"""
Simple FSDP Training Script with Transformers
Uses HuggingFace datasets with caching support.
"""

import os
import time
from datetime import datetime
import multiprocessing
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import functools
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from datasets import load_dataset


def print_timing_box(rank_prefix, title, items):
    """Print timing information in a visually distinct box."""
    prefix_len = len(rank_prefix)
    box_width = 70
    content_width = box_width - prefix_len
    print()  # Empty line before box
    print(f"{rank_prefix}{'=' * content_width}")
    print(f"{rank_prefix}{title:^{content_width}}")
    print(f"{rank_prefix}{'-' * content_width}")
    for item in items:
        # Replace separator placeholder with actual separator
        if item.startswith("---"):
            print(f"{rank_prefix}{'-' * content_width}")
        else:
            print(f"{rank_prefix}{item}")
    print(f"{rank_prefix}{'=' * content_width}")
    print()  # Empty line after box


class TextDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets with tokenization and caching."""

    def __init__(self, dataset_name="Salesforce/wikitext", dataset_config="wikitext-103-raw-v1", 
                 seq_length=512, cache_dir=None, rank=None):
        """
        Initialize dataset with caching support.
        
        Args:
            dataset_name: Name of the HuggingFace dataset (default: 'Salesforce/wikitext')
            dataset_config: Configuration name for the dataset (default: 'wikitext-103-raw-v1')
            seq_length: Maximum sequence length for tokenization
            cache_dir: Directory for caching datasets (None uses default ~/.cache/huggingface)
            rank: Process rank for logging (optional)
        """
        self.seq_length = seq_length
        self.rank = rank
        rank_prefix = f"[Rank {rank}] " if rank is not None else ""
        
        init_start_time = time.time()
        
        # Load tokenizer
        tokenizer_start_time = time.time()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenizer_time = time.time() - tokenizer_start_time
        
        # Load dataset with caching enabled
        # HuggingFace datasets automatically cache downloaded and processed datasets
        print(f"{rank_prefix}Loading dataset {dataset_name}/{dataset_config}...")
        print(f"{rank_prefix}Cache directory: {cache_dir or 'default (~/.cache/huggingface)'}")
        
        dataset_load_start_time = time.time()
        dataset = load_dataset(
            dataset_name, 
            dataset_config,
            cache_dir=cache_dir,
            split="train"
        )
        dataset_load_time = time.time() - dataset_load_start_time
        
        # Tokenize the dataset using all available processors
        # This will be cached automatically by HuggingFace datasets
        num_proc = os.cpu_count() or multiprocessing.cpu_count()
        print(f"{rank_prefix}Tokenizing dataset using {num_proc} processors (this will be cached for future runs)...")
        tokenize_start_time = time.time()
        self.tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
            num_proc=num_proc
        )
        tokenize_time = time.time() - tokenize_start_time
        
        total_init_time = time.time() - init_start_time
        
        # Print timing information in a box
        timing_items = [
            f"Tokenizer loading:        {tokenizer_time:>8.2f}s",
            f"Dataset load/download:    {dataset_load_time:>8.2f}s ({len(dataset):,} raw examples)",
            f"Tokenization ({num_proc} procs):  {tokenize_time:>8.2f}s",
            "---",  # Separator line
            f"Total initialization:   {total_init_time:>8.2f}s",
            f"Final dataset size:      {len(self.tokenized_dataset):,} examples"
        ]
        print_timing_box(rank_prefix, "DATASET LOADING TIMING", timing_items)

    def _tokenize_function(self, examples):
        """Tokenize text examples."""
        # Tokenize and truncate/pad to seq_length
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors=None  # Return lists, not tensors
        )
        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
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

    # Create dataset and dataloader with caching support
    # Set cache_dir to a shared location for multi-node setups, or None for default
    print(f"[Rank {rank}] Starting dataset initialization...")
    dataset_start_time = time.time()
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    dataset = TextDataset(
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-103-raw-v1",
        seq_length=512,
        cache_dir=cache_dir,
        rank=rank
    )
    dataset_init_time = time.time() - dataset_start_time
    
    sampler_start_time = time.time()
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    sampler_time = time.time() - sampler_start_time

    dataloader_start_time = time.time()
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Per-device batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    dataloader_time = time.time() - dataloader_start_time
    
    # Print dataloader timing in a box
    rank_prefix = f"[Rank {rank}] "
    dataloader_items = [
        f"Dataset creation:         {dataset_init_time:>8.2f}s",
        f"Sampler creation:         {sampler_time:>8.3f}s",
        f"DataLoader creation:     {dataloader_time:>8.3f}s",
        "---",  # Separator line
        f"Total setup time:        {dataset_init_time + sampler_time + dataloader_time:>8.2f}s"
    ]
    print_timing_box(rank_prefix, "DATALOADER SETUP TIMING", dataloader_items)

    # Training loop
    num_epochs = 3000
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
