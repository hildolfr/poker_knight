# CUDA/GPU Acceleration Documentation

This directory contains documentation related to Poker Knight's CUDA GPU acceleration implementation.

## Files

- **CUDA_STATUS.md** - Current status of CUDA implementation
- **CUDA_IMPLEMENTATION_SUMMARY.md** - Technical implementation details
- **CUDA_WORKING_SUMMARY.md** - Summary of working GPU acceleration
- **CUDA_FINAL_SUMMARY.md** - Final implementation summary with performance metrics
- **CUDA_TEST_FIXES.md** - Documentation of test fixes for CUDA integration
- **monte_carlo_gpu_analysis.md** - Analysis of Monte Carlo GPU implementation

## Key Points

- GPU acceleration provides up to 1700x speedup (13ms vs 22s for 1M simulations)
- Automatic fallback to CPU when GPU is not available
- Configurable via `config.json` settings
- Full test coverage maintained with GPU implementation