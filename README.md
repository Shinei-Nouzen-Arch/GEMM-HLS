# GEMM-Systolic: FPGA-Accelerated General Matrix Multiplication

A high-performance **General Matrix Multiplication (GEMM)** accelerator implemented in HLS C++ for Xilinx FPGAs. Features a **32x32 systolic array** architecture with Row-Stationary dataflow, targeting **300+ MHz** on Alveo U50.

```
C[M x N] = A[M x K] x B[K x N]    (FP32, M/N/K must be multiples of 32)
```

## Architecture

```
                    DRAM (512-bit AXI)
                   /        |        \
            Read A(1x)   Read B    Write C
               |            |         ^
               v            v         |
          ┌─────────┐  ┌─────────┐   |
          │  URAM   │  │  BRAM   │   |
          │ A Cache │  │ B Buf   │   |
          └────┬────┘  └────┬────┘   |
               └──────┬─────┘        |
                      v              |
              ┌───────────────┐      |
              │  32x32 FP32   │      |
              │ Systolic Array│──────┘
              │  (1024 MACs)  │
              └───────────────┘
```

### Key Optimizations

| Technique | Benefit |
|-----------|---------|
| **32x32 Systolic Array** | 1024 parallel MACs, regular dataflow |
| **Row-Stationary Scheduling** | A matrix read from DRAM only once per row block, cached in URAM |
| **DATAFLOW Pipeline** | Read / Compute / Write tasks overlap in parallel |
| **512-bit AXI Interface** | 16 floats per transaction (64 bytes/cycle) |
| **Chunked Accumulators** | 4 x 256-bit chunks for efficient BRAM mapping |
| **Diagonal Feeding** | Staggered input timing for systolic wavefront propagation |

### Theoretical Peak Performance

```
1024 MACs x 300 MHz = 307.2 GFLOPS (FP32)
```

## File Structure

```
.
├── types.h               # Hardware config, data types, HLS pragma macros
├── systolic_row.h        # Single systolic row module (32 PEs with MAC units)
├── gemm_general.h        # SystolicEngine: 32x32 array orchestrator
├── gemm_general.cpp      # Top-level IP core with DATAFLOW optimization
├── gemm_general_tb.cpp   # Testbench with verification & performance benchmarking
└── README.md
```

### Module Hierarchy

```
GEMM_General (top-level, AXI interface)
├── Task_Read_A_Row          → DRAM → URAM cache (one-time read)
├── process_one_C_tile       → DATAFLOW pipeline per C tile
│   ├── Task_Read_B_All      → Read B from DRAM via stream
│   ├── Task_Compute_All     → Systolic computation (A from URAM, B from stream)
│   │   └── SystolicEngine   → 32x32 array controller
│   │       └── SystolicRow[32]  → 32 PEs each, MAC + accumulate
│   └── Task_Write_C_All     → Write C results back to DRAM
└── Main loop: Row-Stationary scheduling over (M/32) x (N/32) tiles
```

## Hardware Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Array Size | 32 x 32 | Systolic array dimensions |
| Data Type | FP32 | Single-precision floating point |
| AXI Width | 512-bit | 16 floats per AXI transaction |
| Max K | 4096 | Maximum supported K dimension |
| Target Device | Alveo U50 | Xilinx FPGA accelerator card |
| Target Frequency | 300+ MHz | Clock frequency |

## Testing

The testbench (`gemm_general_tb.cpp`) includes **7 test cases** with automatic verification:

| Test Case | Dimensions | Description |
|-----------|------------|-------------|
| `test_32x32_basic` | 32×32×32 | Single tile, random values |
| `test_64x64` | 64×64×64 | 2×2 tiles, random values |
| `test_non_square` | 64×128×96 | Non-square matrices |
| `test_identity` | 64×64×64 | Identity matrix (A×I = A) |
| `test_constant` | 64×64×64 | Constant matrices |
| `test_large` | 128×128×128 | Large matrix (4×4 tiles) |
| `test_boundary` | 32×32×4096 | Maximum K dimension |

**Verification method**: Compare hardware results against software reference implementation with tolerance `1e-3`.

## Build & Run

### Prerequisites

- Xilinx Vitis HLS 2022.2+ (or Vivado HLS)
- Alveo U50 platform files (for hardware build)

### C-Simulation

```bash
# In Vitis HLS GUI or command line:
vitis_hls -f run_csim.tcl

# Or manually:
# 1. Create a Vitis HLS project
# 2. Add gemm_general.cpp as source, gemm_general_tb.cpp as testbench
# 3. Set GEMM_General as top function
# 4. Run C Simulation
```

**Expected output:**
```
============================================
  GEMM_General Testbench (Optimized)
============================================
Passed: 7 / 7

PERFORMANCE SUMMARY
Test                      M×N×K    Time(ms)    GFLOPS
------------------------------------------------------------
test_32x32_basic         32x 32x 32      X.XXX      XXX.XXX
test_64x64               64x 64x 64      X.XXX      XXX.XXX
...
All tests PASSED!
```

### Synthesis

```bash
# In Vitis HLS:
# 1. Set target device to xcu50-fsvh2104-2-e (Alveo U50)
# 2. Set clock period to 3.33ns (300 MHz)
# 3. Run C Synthesis
```

### Hardware Build (Vitis)

```bash
v++ -t hw --platform xilinx_u50_gen3x16_xdma_5_202210_1 \
    -c -k GEMM_General gemm_general.cpp -o gemm_general.xo

v++ -t hw --platform xilinx_u50_gen3x16_xdma_5_202210_1 \
    -l gemm_general.xo -o gemm_general.xclbin
```

## Design Details

### Memory Hierarchy

| Level | Storage | Content | Access Pattern |
|-------|---------|---------|----------------|
| DRAM | HBM/DDR | Full A, B, C matrices | Burst via 512-bit AXI |
| URAM | On-chip | A row block cache (32 x K) | Read once, reuse N/32 times |
| BRAM | On-chip | Tile buffers, accumulators | Per-cycle random access |
| Registers | FF | PE local a/b registers | Shift every cycle |

### Matrix Tiling Strategy

For arbitrary large matrices, the design uses **3-level nested tiling** to fit computation into the 32x32 systolic array:

```
Input: A[M x K], B[K x N], Output: C[M x N]

Tile dimensions:
- num_tiles_m = M / 32      (row blocks of A and C)
- num_tiles_n = N / 32      (column blocks of B and C)
- num_tiles_k = K / 32      (depth blocks for partial sums)
```

**Example**: For 1024x1024 matrices:
- 32 x 32 x 32 = 32,768 tiles total
- Each C tile requires 32 K-tiles of partial sum accumulation

### Computation Flow

**Three-level nested loop structure:**

```
for tm in [0, M/32):                    # Iterate over row blocks of A
    Load A[tm*32 : (tm+1)*32, 0:K] → URAM cache (read once from DRAM)

    for tn in [0, N/32):                # Iterate over column blocks of B
        Clear accumulators

        for tk in [0, K/32):            # Iterate over depth (K dimension)
            Read A_tile[32x32] from URAM at offset tk*32
            Read B_tile[32x32] from DRAM at B[tk*32:(tk+1)*32, tn*32:(tn+1)*32]

            Feed into systolic array with diagonal skew (127 cycles)
            Accumulate partial products into C_tile[32x32]

        Drain final C_tile[32x32] → Write to DRAM at C[tm*32:(tm+1)*32, tn*32:(tn+1)*32]
```

**Key insight**: A row block is cached in URAM and reused N/32 times (once per column block), reducing DRAM reads by 32x.

### Address Calculation for Tiles

```cpp
// A matrix: row-major layout
A_tile_addr = (tm * 32 * K) + (tk * 32)           // Start of A[tm, tk] tile

// B matrix: row-major layout
B_tile_addr = (tk * 32 * N) + (tn * 32)           // Start of B[tk, tn] tile

// C matrix: row-major layout
C_tile_addr = (tm * 32 * N) + (tn * 32)           // Start of C[tm, tn] tile
```

Each 32x32 tile is transferred via **64 AXI transactions** (32 rows × 2 transactions per row, since 512-bit = 16 floats).

### Constraints

- M, N, K must be **multiples of 32**
- K <= 4096 (configurable via `MAX_K` in `types.h`)
- FP32 only (single-precision)
