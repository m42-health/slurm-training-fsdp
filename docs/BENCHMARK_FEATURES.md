# Comprehensive Benchmark Suite - Features Overview

This document describes the comprehensive benchmark suite added to the SLURM training test script.

## What Was Added

### 1. **benchmark.py** - Comprehensive Benchmark Suite (600+ lines)
   
A full-featured benchmarking script that tests all major cluster resources:

#### Storage I/O Tests
- ✅ **Sequential Write**: Tests filesystem write bandwidth with large blocks
- ✅ **Sequential Read**: Tests filesystem read bandwidth  
- ✅ **Parallel Write**: Multi-process concurrent write test (8-16 workers)
- ✅ **Parallel Read**: Multi-process concurrent read test (8-16 workers)
- ✅ **Random IOPS**: 4K block random I/O operations per second

**Purpose**: Identify storage bottlenecks that affect data loading pipelines

#### CPU Tests
- ✅ **Multi-core Stress**: All cores running intensive computation
- ✅ **Matrix Operations**: NumPy BLAS performance (GFLOPS)
- ✅ **Compression**: zlib compression throughput
- ✅ **Sorting**: Large-scale NumPy sorting performance

**Purpose**: Test CPU compute and multi-core scaling outside of GPU workloads

#### Memory Tests
- ✅ **Memory Bandwidth**: Large array copy operations (GB/s)
- ✅ **Allocation Speed**: Repeated allocation/deallocation (ops/s)

**Purpose**: Identify RAM subsystem performance and bottlenecks

#### GPU Tests (per GPU)
- ✅ **Memory Bandwidth**: GPU VRAM copy performance (GB/s)
- ✅ **Compute Throughput**: FP32 matrix multiplication (TFLOPS)
- ✅ **All-Reduce**: Distributed collective operation bandwidth (multi-node)

**Purpose**: Baseline GPU performance and detect issues before training

#### Network Tests (multi-node only)
- ✅ **Node-to-node Latency**: Ping-pong test (microseconds)
- ✅ **Node-to-node Bandwidth**: Large transfer throughput (GB/s)

**Purpose**: Critical for distributed training efficiency

### 2. **Updated slurm_submit.sh**

Modified to run in two phases:
- **Phase 1**: Comprehensive benchmarks (all nodes)
- **Phase 2**: Training (as before)

Both phases use the same distributed setup, so network and GPU collective operations are properly tested.

### 3. **benchmark_only.sh** - Standalone Benchmark Job

A separate SLURM script for running just the benchmarks without training. Useful for:
- Initial cluster setup validation
- Periodic cluster health checks
- Troubleshooting performance issues
- Comparing different node configurations

### 4. **Updated README.md**

Complete documentation including:
- How benchmarks work
- How to interpret results
- Expected performance ranges for common hardware
- How to run benchmarks separately
- Monitoring and results access

### 5. **Results Output**

Benchmarks produce two types of output:

1. **Console Output**: 
   - Formatted tables with all benchmark results
   - Saved in `logs/train_JOBID.out` (with training)
   - Saved in `logs/benchmark_JOBID.out` (benchmark-only)

2. **JSON Results**: 
   - Machine-readable results in `logs/benchmark_JOBID.json`
   - Includes timestamp, hostname, all metrics
   - Easy to parse for automated analysis

## Usage Examples

### Full Training Run (Benchmarks + Training)
```bash
sbatch slurm_submit.sh
# Watch output
tail -f logs/train_*.out
# View JSON results
cat logs/benchmark_*.json
```

### Benchmark Only
```bash
sbatch benchmark_only.sh
# Watch output
tail -f logs/benchmark_*.out
```

### Local Testing (No SLURM)
```bash
source .venv/bin/activate
python benchmark.py
```

## Benchmark Output Example

```
================================================================================
                        BENCHMARK RESULTS
================================================================================

STORAGE I/O
--------------------------------------------------------------------------------
  Sequential Write                                         1234.56 MB/s
  Sequential Read                                          2345.67 MB/s
  Parallel Write (8 workers)                               3456.78 MB/s
  Parallel Read (8 workers)                                4567.89 MB/s
  Random IOPS (4K blocks)                                  12345.67 ops/s

CPU
--------------------------------------------------------------------------------
  Multi-core stress (all cores)                            1234.56 K ops/s
  Matrix multiply (NumPy)                                  123.45 GFLOPS
  Compression (zlib)                                       234.56 MB/s
  Sorting (NumPy)                                          12.34 M elements/s

MEMORY
--------------------------------------------------------------------------------
  Bandwidth (copy)                                         45.67 GB/s
  Allocation/deallocation                                  1234.56 ops/s

GPU 0
--------------------------------------------------------------------------------
  Memory bandwidth                                         1234.56 GB/s
  Compute (FP32 matmul)                                    123.45 TFLOPS

[... more GPUs ...]

NETWORK
--------------------------------------------------------------------------------
  Node-to-node latency                                     1.23 ms
  Node-to-node bandwidth                                   12.34 GB/s

================================================================================
Total benchmark time: 180.45 seconds
================================================================================
```

## Performance Expectations

### Storage
- **NVMe SSD**: 3-7 GB/s sequential, 500K+ IOPS
- **SATA SSD**: 0.5-1 GB/s sequential, 100K IOPS
- **HDD**: 0.1-0.2 GB/s sequential, <200 IOPS
- **Network FS (Lustre/NFS)**: Varies widely, 1-10 GB/s aggregate

### CPU
- **Modern Xeon/EPYC**: 50-200+ GFLOPS (depends on BLAS library)
- **Core count scaling**: Should scale near-linearly for stress test

### Memory
- **DDR4**: 20-40 GB/s per channel
- **DDR5**: 40-80 GB/s per channel
- **Multi-channel**: Multiply by number of channels (typically 4-8)

### GPU (examples)
- **A100 40GB**: ~1500 GB/s memory, ~150-312 TFLOPS (FP32/TF32)
- **A100 80GB**: ~2000 GB/s memory, ~150-312 TFLOPS (FP32/TF32)
- **H100**: ~3000 GB/s memory, ~500+ TFLOPS (FP32)
- **V100**: ~900 GB/s memory, ~14 TFLOPS (FP32)

### Network
- **InfiniBand EDR**: ~12 GB/s, <2μs latency
- **InfiniBand HDR**: ~25 GB/s, <1μs latency
- **100G Ethernet**: ~10-12 GB/s, 5-10μs latency
- **10G Ethernet**: ~1 GB/s, 10-50μs latency

## Why This Matters

1. **Early Problem Detection**: Find hardware/configuration issues before wasting time on failed training runs

2. **Performance Baseline**: Establish what "normal" looks like for your cluster

3. **Troubleshooting**: When training is slow, benchmarks help identify the bottleneck

4. **Cluster Comparison**: Compare different node types or configurations objectively

5. **Verification**: Ensure cluster is performing as expected after maintenance/updates

6. **Resource Planning**: Understand which resources are limiting factors for your workload

## Technical Details

### Parallel Execution
- Storage tests use Python multiprocessing
- CPU tests utilize all available cores
- GPU tests run on all available devices
- Network tests use PyTorch distributed (when available)

### Safe Defaults
- Uses temporary files (auto-cleanup)
- Respects system resources
- No destructive operations
- Handles missing features gracefully

### Distributed Aware
- Works in single-node mode (basic tests)
- Enables network tests in multi-node mode
- Uses same distributed setup as training
- Properly synchronized across all ranks

## Dependencies

All required packages are already in `requirements.txt`:
- `torch>=2.1.0` - For GPU tests and distributed operations
- `numpy>=1.24.0` - For CPU and memory tests
- Python standard library - For I/O and system tests

No additional dependencies required!

## Future Enhancements (Optional Ideas)

- [ ] GPU peer-to-peer (P2P) bandwidth tests
- [ ] Memory access pattern tests (stride/random)
- [ ] MPI collective operation benchmarks
- [ ] Sustained long-duration stress tests
- [ ] Automatic performance regression detection
- [ ] HTML report generation
- [ ] Integration with monitoring systems (Prometheus, etc.)
- [ ] Comparison with previous runs
- [ ] Per-node detailed reports in multi-node setups

