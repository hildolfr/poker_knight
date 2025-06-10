# CUDA/GPU Acceleration Status

## Current State: Experimental (Disabled by Default)

The CUDA GPU acceleration infrastructure has been implemented but is currently **disabled by default** due to accuracy issues in the simplified kernel implementation.

## What's Working

✅ **Infrastructure**
- GPU detection and device information retrieval
- CUDA kernel compilation system with caching
- Memory management and GPU resource allocation
- Automatic CPU fallback on GPU failure
- GPU usage reporting in results

✅ **Integration**
- Seamless integration with main solver
- Configuration-based GPU control
- Support for `always_use_gpu` option
- Proper error handling and fallback

## What Needs Work

❌ **Kernel Accuracy**
- Current simplified kernel produces incorrect poker hand evaluations
- Win probabilities are not matching CPU implementation
- Proper Monte Carlo simulation logic needs to be implemented

❌ **Performance**
- No performance benefit yet due to simplified kernel
- Proper hand evaluation lookup tables need implementation
- SIMD optimizations not yet utilized

## Technical Details

### Architecture
```
poker_knight/cuda/
├── __init__.py          # GPU detection and configuration
├── gpu_solver.py        # High-level GPU solver interface
├── kernels.py           # Kernel compilation and management
├── kernels/
│   └── monte_carlo.cu   # CUDA kernel source (not used currently)
├── lookup_tables.py     # Lookup table generation
├── benchmarks.py        # Performance benchmarking
└── tests/               # Unit tests
```

### Key Components

1. **GPU Solver** (`gpu_solver.py`)
   - Converts card strings to GPU format
   - Manages kernel execution
   - Returns results in standard format

2. **Kernel Manager** (`kernels.py`)
   - Compiles CUDA kernels with caching
   - Provides Python interface to kernels
   - Falls back to simplified kernels

3. **Memory Format**
   - Cards: 8-bit packed format (4 bits rank, 2 bits suit, 2 bits flags)
   - Results: Atomic accumulation in global memory

## How to Enable GPU (Experimental)

1. Edit `poker_knight/config.json`:
   ```json
   "cuda_settings": {
       "enable_cuda": true,
       ...
   }
   ```

2. Ensure CuPy is installed:
   ```bash
   pip install cupy-cuda11x  # or appropriate version
   ```

3. Run with GPU acceleration:
   ```python
   from poker_knight import solve_poker_hand
   result = solve_poker_hand(['A♠', 'K♠'], 2)
   print(f"GPU used: {result.gpu_used}")
   ```

## Future Work

1. **Implement Accurate Kernel**
   - Port full hand evaluation logic to CUDA
   - Implement proper deck dealing and card removal
   - Add board card support

2. **Optimize Performance**
   - Use texture memory for lookup tables
   - Implement warp-level primitives
   - Optimize memory access patterns

3. **Extended Features**
   - Multi-GPU support
   - Batch processing for multiple hands
   - Hand category tracking on GPU

## Testing

To test GPU functionality:
```bash
python test_gpu_minimal.py      # Basic GPU tests
python test_cuda_direct.py      # Direct kernel test
python examples/simple_gpu_demo.py  # User-facing demo
```

## Known Issues

1. Kernel produces incorrect win probabilities
2. Cache serialization warning (cosmetic)
3. No performance benefit in current state

## Contributing

If you're interested in improving GPU support:
1. Focus on implementing accurate hand evaluation in CUDA
2. Use the existing test framework to validate results
3. Ensure CPU fallback continues to work