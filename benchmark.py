#!/usr/bin/env python3
"""
Comprehensive SLURM Cluster Benchmark Suite
Tests storage I/O, CPU, GPU, network, and memory performance
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import socket
import numpy as np

try:
    import torch
    import torch.distributed as dist

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available, skipping GPU benchmarks")


class BenchmarkResults:
    """Store and display benchmark results"""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

    def add(self, category, test_name, value, unit=""):
        if category not in self.results:
            self.results[category] = {}
        # Format value to 2 decimal places and combine with unit
        if isinstance(value, float):
            formatted = f"{value:.2f} {unit}".strip()
        else:
            formatted = f"{value} {unit}".strip()
        self.results[category][test_name] = formatted

    def print_summary(self):
        print("\n" + "=" * 80)
        print(f"{'BENCHMARK RESULTS':^80}")
        print("=" * 80)

        for category, tests in self.results.items():
            print(f"\n{category.upper()}")
            print("-" * 80)
            for test_name, value_str in tests.items():
                print(f"  {test_name:<50} {value_str:>25}")

        elapsed = time.time() - self.start_time
        print("\n" + "=" * 80)
        print(f"Total benchmark time: {elapsed:.2f} seconds")
        print("=" * 80 + "\n")

    def save_yaml(self, filepath):
        """Save results to YAML file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "total_time": f"{time.time() - self.start_time:.2f} seconds",
            "results": self.results,
        }

        # Simple YAML writer (no external dependency)
        with open(filepath, "w") as f:
            for key, value in data.items():
                if key == "results":
                    f.write(f"{key}:\n")
                    for category, tests in value.items():
                        f.write(f"  {category}:\n")
                        for test_name, test_value in tests.items():
                            f.write(f"    {test_name}: {test_value}\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"Results saved to: {filepath}")


# ============================================================================
# STORAGE I/O BENCHMARKS
# ============================================================================


def benchmark_storage_sequential_write(size_mb=1024, block_size_kb=4096):
    """Test sequential write performance"""
    print(f"  → Sequential write test ({size_mb} MB, {block_size_kb} KB blocks)...")

    cache_dir = Path(".cache")
    tmp_path = cache_dir / f"bench_seq_write_{os.getpid()}.tmp"

    try:
        data = os.urandom(block_size_kb * 1024)
        blocks = size_mb * 1024 // block_size_kb

        start = time.time()
        with open(tmp_path, "wb") as f:
            for _ in range(blocks):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.time() - start

        throughput = size_mb / elapsed
        return throughput
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def benchmark_storage_sequential_read(size_mb=1024, block_size_kb=4096):
    """Test sequential read performance"""
    print(f"  → Sequential read test ({size_mb} MB, {block_size_kb} KB blocks)...")

    cache_dir = Path(".cache")
    tmp_path = cache_dir / f"bench_seq_read_{os.getpid()}.tmp"

    data = os.urandom(block_size_kb * 1024)
    blocks = size_mb * 1024 // block_size_kb

    # Write test file
    with open(tmp_path, "wb") as f:
        for _ in range(blocks):
            f.write(data)

    try:
        start = time.time()
        with open(tmp_path, "rb") as f:
            while f.read(block_size_kb * 1024):
                pass
        elapsed = time.time() - start

        throughput = size_mb / elapsed
        return throughput
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def worker_write(args):
    """Worker function for parallel write test"""
    worker_id, size_mb, block_size_kb = args
    cache_dir = Path(".cache")
    tmp_path = cache_dir / f"bench_w{worker_id}_{os.getpid()}.tmp"

    try:
        data = os.urandom(block_size_kb * 1024)
        blocks = size_mb * 1024 // block_size_kb

        start = time.time()
        with open(tmp_path, "wb") as f:
            for _ in range(blocks):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.time() - start

        return size_mb / elapsed
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def benchmark_storage_parallel_write(num_workers=8, size_mb_per_worker=256):
    """Test parallel write performance with multiple processes"""
    print(
        f"  → Parallel write test ({num_workers} workers, {size_mb_per_worker} MB each)..."
    )

    args = [(i, size_mb_per_worker, 4096) for i in range(num_workers)]

    start = time.time()
    with mp.Pool(num_workers) as pool:
        throughputs = pool.map(worker_write, args)
    elapsed = time.time() - start

    total_mb = num_workers * size_mb_per_worker
    aggregate_throughput = total_mb / elapsed

    return aggregate_throughput


def worker_read(args):
    """Worker function for parallel read test"""
    worker_id, tmp_path, size_mb, block_size_kb = args

    start = time.time()
    with open(tmp_path, "rb") as f:
        while f.read(block_size_kb * 1024):
            pass
    elapsed = time.time() - start

    return size_mb / elapsed


def benchmark_storage_parallel_read(num_workers=8, size_mb_per_worker=256):
    """Test parallel read performance with multiple processes"""
    print(
        f"  → Parallel read test ({num_workers} workers, {size_mb_per_worker} MB each)..."
    )

    # Create test files
    cache_dir = Path(".cache")
    tmp_paths = []
    block_size_kb = 4096
    data = os.urandom(block_size_kb * 1024)
    blocks = size_mb_per_worker * 1024 // block_size_kb

    for i in range(num_workers):
        tmp_path = cache_dir / f"bench_r{i}_{os.getpid()}.tmp"
        tmp_paths.append(tmp_path)
        with open(tmp_path, "wb") as f:
            for _ in range(blocks):
                f.write(data)

    try:
        args = [
            (i, tmp_paths[i], size_mb_per_worker, block_size_kb)
            for i in range(num_workers)
        ]

        start = time.time()
        with mp.Pool(num_workers) as pool:
            throughputs = pool.map(worker_read, args)
        elapsed = time.time() - start

        total_mb = num_workers * size_mb_per_worker
        aggregate_throughput = total_mb / elapsed

        return aggregate_throughput
    finally:
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def benchmark_storage_random_iops(num_ops=10000, file_size_mb=100):
    """Test random I/O operations per second"""
    print(f"  → Random IOPS test ({num_ops} operations on {file_size_mb} MB file)...")

    cache_dir = Path(".cache")
    tmp_path = cache_dir / f"bench_random_{os.getpid()}.tmp"

    # Create file with random data
    with open(tmp_path, "wb") as f:
        f.write(os.urandom(file_size_mb * 1024 * 1024))

    try:
        block_size = 4096
        file_size = file_size_mb * 1024 * 1024
        max_offset = file_size - block_size

        start = time.time()
        with open(tmp_path, "rb+") as f:
            for _ in range(num_ops):
                offset = np.random.randint(0, max_offset)
                f.seek(offset)
                f.read(block_size)
        elapsed = time.time() - start

        iops = num_ops / elapsed
        return iops
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def benchmark_multinode_storage_aggregate():
    """Test aggregate storage I/O across all nodes simultaneously"""
    if not HAS_TORCH or not dist.is_initialized():
        return None, None

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        return None, None

    if rank == 0:
        print(
            f"  → Multi-node aggregate storage test ({world_size} nodes simultaneous)..."
        )

    # All nodes write simultaneously
    size_mb = 1024
    cache_dir = Path(".cache")
    tmp_path = cache_dir / f"bench_multinode_{rank}_{os.getpid()}.tmp"

    try:
        data = os.urandom(4096 * 1024)
        blocks = size_mb * 1024 // 4096

        # Barrier to sync all nodes
        dist.barrier()

        start = time.time()
        with open(tmp_path, "wb") as f:
            for _ in range(blocks):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        write_time = time.time() - start

        dist.barrier()

        # All nodes read simultaneously
        start = time.time()
        with open(tmp_path, "rb") as f:
            while f.read(4096 * 1024):
                pass
        read_time = time.time() - start

        dist.barrier()

        # Gather times from all ranks to compute aggregate
        # Use GPU tensors if NCCL backend is used (NCCL doesn't support CPU)
        device = (
            torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        write_tensor = torch.tensor([write_time], dtype=torch.float32, device=device)
        read_tensor = torch.tensor([read_time], dtype=torch.float32, device=device)

        if rank == 0:
            write_times = [torch.zeros(1, device=device) for _ in range(world_size)]
            read_times = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.gather(write_tensor, write_times, dst=0)
            dist.gather(read_tensor, read_times, dst=0)

            # Aggregate throughput = total data / max time
            max_write_time = max(t.item() for t in write_times)
            max_read_time = max(t.item() for t in read_times)

            aggregate_write_bw = (size_mb * world_size) / max_write_time
            aggregate_read_bw = (size_mb * world_size) / max_read_time

            return aggregate_write_bw, aggregate_read_bw
        else:
            dist.gather(write_tensor, dst=0)
            dist.gather(read_tensor, dst=0)
            return None, None

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_storage_benchmarks(results):
    """Run all storage I/O benchmarks"""
    print("\n[1/5] STORAGE I/O BENCHMARKS")
    print("=" * 80)

    # Per-node operations
    write_bw = benchmark_storage_sequential_write(size_mb=1024, block_size_kb=4096)
    results.add("Storage I/O (per-node)", "Sequential Write", write_bw, "MB/s")

    read_bw = benchmark_storage_sequential_read(size_mb=1024, block_size_kb=4096)
    results.add("Storage I/O (per-node)", "Sequential Read", read_bw, "MB/s")

    # Parallel operations
    num_workers = min(mp.cpu_count(), 16)

    par_write_bw = benchmark_storage_parallel_write(
        num_workers=num_workers, size_mb_per_worker=256
    )
    results.add(
        "Storage I/O (per-node)",
        f"Parallel Write ({num_workers} workers)",
        par_write_bw,
        "MB/s",
    )

    par_read_bw = benchmark_storage_parallel_read(
        num_workers=num_workers, size_mb_per_worker=256
    )
    results.add(
        "Storage I/O (per-node)",
        f"Parallel Read ({num_workers} workers)",
        par_read_bw,
        "MB/s",
    )

    # Random IOPS
    iops = benchmark_storage_random_iops(num_ops=5000, file_size_mb=100)
    results.add("Storage I/O (per-node)", "Random IOPS (4K blocks)", iops, "ops/s")

    # Multi-node aggregate tests
    agg_write, agg_read = benchmark_multinode_storage_aggregate()
    if agg_write is not None:
        results.add(
            "Storage I/O (multi-node aggregate)", "Write bandwidth", agg_write, "MB/s"
        )
        results.add(
            "Storage I/O (multi-node aggregate)", "Read bandwidth", agg_read, "MB/s"
        )


# ============================================================================
# CPU BENCHMARKS
# ============================================================================


def cpu_stress_worker(duration_sec):
    """CPU stress worker - intensive computation"""
    end_time = time.time() + duration_sec
    ops = 0
    while time.time() < end_time:
        # CPU-intensive operations
        _ = sum([i**2 for i in range(1000)])
        ops += 1
    return ops


def benchmark_cpu_stress(duration_sec=10):
    """Multi-core CPU stress test"""
    num_cores = mp.cpu_count()
    print(f"  → CPU stress test ({num_cores} cores, {duration_sec}s)...")

    start = time.time()
    with mp.Pool(num_cores) as pool:
        ops_list = pool.map(cpu_stress_worker, [duration_sec] * num_cores)
    elapsed = time.time() - start

    total_ops = sum(ops_list)
    ops_per_sec = total_ops / elapsed

    return ops_per_sec


def benchmark_cpu_matrix_operations(size=2000, iterations=10):
    """Test CPU matrix computation performance"""
    print(f"  → Matrix operations ({size}x{size}, {iterations} iterations)...")

    np.random.seed(42)
    A = np.random.randn(size, size).astype(np.float64)
    B = np.random.randn(size, size).astype(np.float64)

    start = time.time()
    for _ in range(iterations):
        C = np.dot(A, B)
    elapsed = time.time() - start

    # Calculate GFLOPS
    ops_per_matmul = 2 * size**3  # Approximate for matrix multiply
    total_ops = ops_per_matmul * iterations
    gflops = (total_ops / elapsed) / 1e9

    return gflops


def benchmark_cpu_compression(size_mb=100, iterations=5):
    """Test CPU compression performance"""
    print(f"  → Compression test ({size_mb} MB, {iterations} iterations)...")

    import zlib

    data = os.urandom(size_mb * 1024 * 1024)

    start = time.time()
    for _ in range(iterations):
        compressed = zlib.compress(data, level=6)
    elapsed = time.time() - start

    throughput = (size_mb * iterations) / elapsed
    return throughput


def benchmark_cpu_sorting(size_million=10, iterations=5):
    """Test CPU sorting performance"""
    print(f"  → Sorting test ({size_million}M elements, {iterations} iterations)...")

    np.random.seed(42)
    data = np.random.randn(size_million * 1_000_000)

    start = time.time()
    for _ in range(iterations):
        _ = np.sort(data.copy())
    elapsed = time.time() - start

    throughput = (size_million * iterations) / elapsed
    return throughput


def run_cpu_benchmarks(results):
    """Run all CPU benchmarks"""
    print("\n[2/5] CPU BENCHMARKS")
    print("=" * 80)

    # Multi-core stress test
    ops_per_sec = benchmark_cpu_stress(duration_sec=10)
    results.add("CPU", "Multi-core stress (all cores)", ops_per_sec / 1000, "K ops/s")

    # Matrix operations
    gflops = benchmark_cpu_matrix_operations(size=2000, iterations=10)
    results.add("CPU", "Matrix multiply (NumPy)", gflops, "GFLOPS")

    # Compression
    comp_throughput = benchmark_cpu_compression(size_mb=100, iterations=5)
    results.add("CPU", "Compression (zlib)", comp_throughput, "MB/s")

    # Sorting
    sort_throughput = benchmark_cpu_sorting(size_million=10, iterations=5)
    results.add("CPU", "Sorting (NumPy)", sort_throughput, "M elements/s")


# ============================================================================
# MEMORY BENCHMARKS
# ============================================================================


def benchmark_memory_bandwidth(size_gb=2, iterations=5):
    """Test memory bandwidth"""
    print(f"  → Memory bandwidth test ({size_gb} GB, {iterations} iterations)...")

    size = int(size_gb * 1024 * 1024 * 1024 / 8)  # Convert to number of float64s

    # Allocate arrays
    src = np.random.randn(size)

    start = time.time()
    for _ in range(iterations):
        dst = src.copy()
    elapsed = time.time() - start

    # Calculate bandwidth (read + write)
    bytes_per_iter = size * 8 * 2  # 8 bytes per float64, read and write
    total_bytes = bytes_per_iter * iterations
    bandwidth_gbs = (total_bytes / elapsed) / (1024**3)

    return bandwidth_gbs


def benchmark_memory_allocation(num_allocs=1000, size_mb=100):
    """Test memory allocation/deallocation performance"""
    print(f"  → Memory allocation test ({num_allocs} allocations of {size_mb} MB)...")

    size = size_mb * 1024 * 1024 // 8  # Number of float64s

    start = time.time()
    for _ in range(num_allocs):
        arr = np.empty(size, dtype=np.float64)
        del arr
    elapsed = time.time() - start

    allocs_per_sec = num_allocs / elapsed
    return allocs_per_sec


def run_memory_benchmarks(results):
    """Run all memory benchmarks"""
    print("\n[3/5] MEMORY BENCHMARKS")
    print("=" * 80)

    # Bandwidth test
    bandwidth = benchmark_memory_bandwidth(size_gb=2, iterations=5)
    results.add("Memory", "Bandwidth (copy)", bandwidth, "GB/s")

    # Allocation test
    allocs_per_sec = benchmark_memory_allocation(num_allocs=1000, size_mb=100)
    results.add("Memory", "Allocation/deallocation", allocs_per_sec, "ops/s")


# ============================================================================
# GPU BENCHMARKS (PyTorch)
# ============================================================================


def benchmark_gpu_memory_bandwidth(device, size_gb=2, iterations=20):
    """Test GPU memory bandwidth"""
    print(
        f"  → GPU memory bandwidth test on {device} ({size_gb} GB, {iterations} iterations)..."
    )

    size = int(size_gb * 1024 * 1024 * 1024 / 4)  # Number of float32s

    # Allocate tensors
    src = torch.randn(size, device=device)
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(5):
        dst = src.clone()
        del dst
    torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(iterations):
        dst = src.clone()
        del dst
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    # Calculate bandwidth
    bytes_per_iter = size * 4 * 2  # 4 bytes per float32, read and write
    total_bytes = bytes_per_iter * iterations
    bandwidth_gbs = (total_bytes / elapsed) / (1024**3)

    return bandwidth_gbs


def benchmark_gpu_compute(device, size=8192, iterations=100):
    """Test GPU compute throughput with matrix multiplication"""
    print(
        f"  → GPU compute test on {device} ({size}x{size} matmul, {iterations} iterations)..."
    )

    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
        del C
    torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    # Calculate TFLOPS
    ops_per_matmul = 2 * size**3
    total_ops = ops_per_matmul * iterations
    tflops = (total_ops / elapsed) / 1e12

    return tflops


def benchmark_gpu_all_reduce(size_mb=100, iterations=50):
    """Test GPU all-reduce collective performance (inter-GPU/inter-node)"""
    if not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        print(
            f"  → GPU all-reduce test ({world_size} GPUs, {size_mb} MB, {iterations} iterations)..."
        )

    size = size_mb * 1024 * 1024 // 4  # Number of float32s
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    tensor = torch.randn(size, device=device)

    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    # Calculate bandwidth (algorithmic bandwidth)
    # All-reduce logically transfers 2*(N-1)/N of data size
    bytes_per_iter = size_mb * 1024 * 1024 * 2 * (world_size - 1) / world_size
    total_bytes = bytes_per_iter * iterations
    bandwidth_gbs = (total_bytes / elapsed) / (1024**3)

    return bandwidth_gbs


def benchmark_gpu_all_to_all():
    """Test GPU all-to-all communication (inter-GPU/inter-node)"""
    if not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size < 2:
        return None

    if rank == 0:
        print(f"  → GPU all-to-all test ({world_size} GPUs)...")

    size_mb_per_peer = 25  # 25MB per peer
    size = size_mb_per_peer * 1024 * 1024 // 4  # Number of float32s
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Input/output tensors for all-to-all
    input_tensor = torch.randn(size * world_size, device=device)
    output_tensor = torch.zeros(size * world_size, device=device)
    input_splits = [size] * world_size
    output_splits = [size] * world_size

    # Warmup
    for _ in range(5):
        dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)
    torch.cuda.synchronize(device)

    iterations = 20
    start = time.time()
    for _ in range(iterations):
        dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    # Each rank sends size_mb_per_peer to every other rank
    total_data_mb = size_mb_per_peer * world_size * iterations
    bandwidth_gbs = (total_data_mb / elapsed) / 1024

    return bandwidth_gbs


def run_gpu_benchmarks(results):
    """Run all GPU benchmarks"""
    if not HAS_TORCH or not torch.cuda.is_available():
        print("\n[4/5] GPU BENCHMARKS - SKIPPED (no CUDA available)")
        return

    print("\n[4/5] GPU BENCHMARKS")
    print("=" * 80)

    num_gpus = torch.cuda.device_count()
    print(f"  Found {num_gpus} GPU(s)")

    for gpu_id in range(num_gpus):
        device = torch.device(f"cuda:{gpu_id}")
        gpu_name = torch.cuda.get_device_name(gpu_id)

        print(f"\n  GPU {gpu_id}: {gpu_name}")

        # Memory bandwidth
        mem_bw = benchmark_gpu_memory_bandwidth(device, size_gb=2, iterations=20)
        results.add(f"GPU {gpu_id}", "Memory bandwidth", mem_bw, "GB/s")

        # Compute throughput
        tflops = benchmark_gpu_compute(device, size=8192, iterations=100)
        results.add(f"GPU {gpu_id}", "Compute (FP32 matmul)", tflops, "TFLOPS")

    # Collective operations (if distributed is initialized)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        allreduce_bw = benchmark_gpu_all_reduce(size_mb=100, iterations=50)
        if allreduce_bw is not None and rank == 0:
            results.add(
                "Inter-GPU Communication",
                f"All-reduce ({world_size} GPUs)",
                allreduce_bw,
                "GB/s",
            )

        alltoall_bw = benchmark_gpu_all_to_all()
        if alltoall_bw is not None and rank == 0:
            results.add(
                "Inter-GPU Communication",
                f"All-to-all ({world_size} GPUs)",
                alltoall_bw,
                "GB/s",
            )


# ============================================================================
# NETWORK BENCHMARKS
# ============================================================================


def get_network_info():
    """Get network interface information"""
    try:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        return hostname, ip_addr
    except:
        return "unknown", "unknown"


def benchmark_network_latency():
    """Test network latency between nodes (requires distributed setup)"""
    if not HAS_TORCH or not dist.is_initialized():
        return None

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        return None

    if rank == 0:
        print(f"  → Network latency test (ping-pong between nodes)...")

    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    tensor = torch.zeros(1, device=device)

    iterations = 100

    if rank == 0:
        # Warmup
        for _ in range(10):
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)

        # Actual test
        start = time.time()
        for _ in range(iterations):
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        elapsed = time.time() - start

        latency_ms = (elapsed / iterations / 2) * 1000  # Divide by 2 for one-way
        return latency_ms
    elif rank == 1:
        # Warmup
        for _ in range(10):
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)

        # Actual test
        for _ in range(iterations):
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)

    return None


def benchmark_network_bandwidth():
    """Test network bandwidth between nodes (ping-pong)"""
    if not HAS_TORCH or not dist.is_initialized():
        return None

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        return None

    if rank == 0:
        print(f"  → Network bandwidth test (node-to-node, {world_size} nodes)...")

    size_mb = 100
    size = size_mb * 1024 * 1024 // 4  # Number of float32s
    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    tensor = torch.randn(size, device=device)

    iterations = 20

    if rank == 0:
        # Warmup
        for _ in range(5):
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Actual test
        start = time.time()
        for _ in range(iterations):
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate bandwidth (bidirectional)
        bytes_per_iter = size_mb * 1024 * 1024 * 2  # Send and receive
        total_bytes = bytes_per_iter * iterations
        bandwidth_gbs = (total_bytes / elapsed) / (1024**3)
        return bandwidth_gbs
    elif rank == 1:
        # Warmup
        for _ in range(5):
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)

        # Actual test
        for _ in range(iterations):
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)

    return None


def benchmark_network_all_to_all():
    """Test aggregate network bandwidth (all-to-all pattern)"""
    if not HAS_TORCH or not dist.is_initialized():
        return None

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        return None

    if rank == 0:
        print(f"  → Network all-to-all bandwidth test ({world_size} nodes)...")

    # Use GPU tensors (NCCL backend requires it, network is still tested inter-node)
    size_mb_per_peer = 25
    size = size_mb_per_peer * 1024 * 1024 // 4

    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    input_tensor = torch.randn(size * world_size, device=device)
    output_tensor = torch.zeros(size * world_size, device=device)
    input_splits = [size] * world_size
    output_splits = [size] * world_size

    # Warmup
    for _ in range(3):
        dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)

    iterations = 10
    start = time.time()
    for _ in range(iterations):
        dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)
    elapsed = time.time() - start

    # Total data transferred per iteration
    total_data_mb = size_mb_per_peer * world_size * iterations
    bandwidth_gbs = (total_data_mb / elapsed) / 1024

    return bandwidth_gbs


def run_network_benchmarks(results):
    """Run all network benchmarks"""
    print("\n[5/5] NETWORK BENCHMARKS")
    print("=" * 80)

    hostname, ip_addr = get_network_info()
    print(f"  Hostname: {hostname}")
    print(f"  IP Address: {ip_addr}")

    if not HAS_TORCH or not dist.is_initialized():
        print("  → Network tests require distributed setup - SKIPPED")
        results.add("Network", "Status", "Not initialized", "")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"  World size: {world_size} nodes")

    if world_size < 2:
        if rank == 0:
            print("  → Network tests require at least 2 nodes - SKIPPED")
        results.add("Network", "Status", "Single node", "")
        return

    # Latency test
    latency = benchmark_network_latency()
    if latency is not None and rank == 0:
        results.add("Inter-Node Network", "Latency (ping-pong)", latency, "ms")

    # Bandwidth test (point-to-point)
    bandwidth = benchmark_network_bandwidth()
    if bandwidth is not None and rank == 0:
        results.add(
            "Inter-Node Network",
            f"Bandwidth (ping-pong, {world_size} GPUs)",
            bandwidth,
            "GB/s",
        )

    # All-to-all bandwidth (aggregate)
    alltoall_bw = benchmark_network_all_to_all()
    if alltoall_bw is not None and rank == 0:
        results.add(
            "Inter-Node Network",
            f"All-to-all bandwidth ({world_size} GPUs)",
            alltoall_bw,
            "GB/s",
        )


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run all benchmarks"""
    # Initialize distributed if running via torchrun
    if HAS_TORCH and not dist.is_initialized():
        # Check if we're running in a distributed environment
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            try:
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
            except Exception as e:
                print(f"Warning: Could not initialize distributed: {e}")

    # Only rank 0 prints header
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n" + "=" * 80)
        print(f"{'SLURM CLUSTER BENCHMARK SUITE':^80}")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Hostname: {socket.gethostname()}")
        print(f"CPU cores: {mp.cpu_count()}")

        if HAS_TORCH and torch.cuda.is_available():
            print(f"GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("GPUs available: 0")

        if HAS_TORCH and dist.is_initialized():
            print(f"Distributed: rank {dist.get_rank()} of {dist.get_world_size()}")
        else:
            print("Distributed: not initialized")

        print("=" * 80)

    # Initialize results
    results = BenchmarkResults()

    # Create cache directory for benchmark files
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    # Run benchmarks (inter-node tests first for YAML ordering)
    try:
        run_network_benchmarks(results)
        run_gpu_benchmarks(results)
        run_storage_benchmarks(results)
        run_cpu_benchmarks(results)
        run_memory_benchmarks(results)
    except Exception as e:
        print(f"\nERROR during benchmarks: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up cache directory
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir, ignore_errors=True)

    # Print and save results (only rank 0)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            results.print_summary()

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            results_file = log_dir / f"benchmark_{job_id}.yaml"
            results.save_yaml(results_file)
            print(
                f"\nNote: Multi-node aggregate metrics collected from {world_size} nodes"
            )
    else:
        # Non-distributed mode
        results.print_summary()

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        results_file = log_dir / f"benchmark_{job_id}.yaml"
        results.save_yaml(results_file)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up distributed
        if HAS_TORCH and dist.is_initialized():
            dist.destroy_process_group()
