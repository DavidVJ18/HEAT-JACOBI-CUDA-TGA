# 🔥 Heat Equation — Jacobi Solver with CUDA

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-12.2.2-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/GPU-RTX%203080%20%7C%20RTX%204090-green?style=for-the-badge&logo=nvidia" />
  <img src="https://img.shields.io/badge/Arch-sm__86%20%7C%20sm__89-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Cluster-Boada%20%E2%80%93%20SLURM-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Course-TGA%20FIB%20UPC%202025--26-red?style=for-the-badge" />
</p>

> CUDA-accelerated solver for the 2D steady-state heat equation using the Jacobi iterative method. The project explores multiple GPU kernel strategies — row-parallel, column-parallel, and 2D block tiling with shared memory — along with CUDA streams, pinned host memory, two-pass parallel reduction, and multi-GPU scaling up to 4 GPUs via Peer-to-Peer (P2P) access.

**Authors:** David Víctor Juanas & David Castro Paniello  
**Course:** Tarjetas Gráficas y Aceleradores (TGA) — FIB, Universitat Politècnica de Catalunya (UPC), Q1 2025-26  
**Cluster:** Boada (boada-10) — 4× NVIDIA RTX 3080 + 1× NVIDIA RTX 4090

---

## 📋 Table of Contents

- [Problem Description](#%EF%B8%8F-problem-description)
- [Mathematical Background](#-mathematical-background)
- [Repository Structure](#-repository-structure)
- [Single-GPU Implementation](#-single-gpu-implementation-prototipoconjuntocondblock)
- [Multi-GPU Implementation](#-multi-gpu-implementation-prototipovarisgpu)
- [Input File Format](#-input-file-format)
- [Requirements](#-requirements)
- [Build & Run](#-build--run)
- [SLURM Cluster Execution](#-slurm-cluster-execution)
- [Benchmark Results](#-benchmark-results)
- [Documentation](#-documentation)
- [Tech Stack](#-tech-stack)
- [Authors](#-authors)

---

## 🌡️ Problem Description

This project solves the **2D steady-state heat equation** on a square domain (a metal plate) discretized as an `(N+2) × (N+2)` grid. The simulation is initialized with configurable **heat sources**, each defined by a 2D position, a radius of influence, and a temperature value applied to boundary cells via distance-weighted contribution.

- **Border cells** accumulate temperature from nearby heat sources (computed once at initialization).
- **Interior cells** are updated iteratively using the Jacobi stencil until convergence.

---

## 📐 Mathematical Background

The solver targets the **Laplace equation** in 2D steady state:

```
∂²T/∂x² + ∂²T/∂y² = 0
```

Discretized with finite differences, the **Jacobi update** for each interior cell `(i, j)` is:

```
T_new[i][j] = 0.25 * (T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1])
```

Convergence is measured via the **squared residual norm** accumulated across all interior cells:

```
residual = Σ (T_new[i][j] - T[i][j])²
```

The solver iterates until `residual < threshold` or `maxiter` is reached.

**Why is Jacobi ideal for GPU parallelization?**  
Each interior cell can be updated independently within a single iteration — no data dependencies exist between cells at the same time step. This embarrassingly parallel structure maps naturally onto GPU architectures where thousands of threads execute simultaneously.

---

## 📁 Repository Structure

```
HEAT-JACOBI-CUDA-TGA/
│
├── PrototipoConjuntoConBlock/      # Single-GPU implementation
│   ├── heat.cu                     # Main program: argument parsing, solver loop, timing
│   ├── solver.cu                   # All CUDA kernels + solve_cuda() host interface
│   ├── misc.cu                     # Helpers: initialize(), finalize(), write_image(), read_input()
│   ├── heat.h                      # Shared types: algoparam_t, heatsrc_t, declarations
│   ├── Makefile                    # Build script (nvcc, CUDA 12.2.2, sm_86/sm_89)
│   ├── job.sh                      # SLURM job script (RTX 3080 / RTX 4090 options)
│   └── test.dat                    # Example input: 2 heat sources
│
├── PrototipoVariasGPU/             # Multi-GPU implementation (up to 4 GPUs)
│   ├── heat.cu                     # Main program with multi-GPU orchestration
│   ├── solver.cu                   # solve_cuda_multi(), halo_exchange_multi(), swap_mats_dev_multi()
│   ├── misc.cu                     # Shared helpers (same as single-GPU)
│   ├── heat.h                      # Extended types: MAX_GPUS=8, per-GPU arrays
│   ├── Makefile                    # Build script
│   ├── job.sh                      # SLURM job script (4× RTX 3080, CUDA_VISIBLE_DEVICES)
│   └── test.dat                    # Example input
│
├── Memoria_Práctica_TGA.pdf        # Full technical report (design, benchmarks, analysis)
├── PracticasTGA.pdf                # Original assignment statement (FIB UPC)
└── README.md                       # This file
```

---

## ⚙️ Single-GPU Implementation (`PrototipoConjuntoConBlock`)

### Kernel Strategies

The active kernel is selected at **compile time** via macros in `solver.cu`:

| Macro (`THREADWORK`) | Strategy | Thread assignment |
|---|---|---|
| `ROW_WORK` (0) | Row-parallel | One thread per row; iterates over all columns |
| `COLUMN_WORK` (1) | Column-parallel | One thread per column; iterates over all rows |
| `BLOCK_WORK` (2) ✅ | **2D block tiling** | **Default.** One thread per cell, 16×16 blocks |

The `SHARED` macro (0/1) enables shared memory variants for the row/column kernels.

### Block Kernel — `jacobi_shared_block_kernel` (Default)

Uses **2D thread blocks of 16×16** (`BLOCK_SIZE_BLOCK_KERNEL_X/Y = 16`):

1. Each block cooperatively loads its 16×16 tile plus a **1-cell halo** (18×18 total) into shared memory.
2. All Jacobi updates are computed from shared memory — no redundant global memory reads.
3. An **intra-block tree reduction** accumulates the partial squared residual.
4. Thread 0 of each block writes one `double` to `diffsBlock_dev`.

### Two-Pass Parallel Reduction

After each Jacobi kernel launch, `reduction_kernel07` aggregates all per-block residuals:

- **Pass 1:** Massively parallel reduction into a vector of partial sums (up to 1024 blocks × 256 threads, with sequential addressing and loop unrolling).
- **Pass 2:** Single block reduces the partial sums to the final scalar residual, copied to the host.

This design requires only **2 kernel launches** per iteration for convergence checking.

### CUDA Streams

When streams are enabled, the grid of block-rows is partitioned among `nStreams` CUDA streams. Each stream launches `jacobi_shared_block_kernel` on its assigned horizontal slice using a `by_offset` parameter. All streams synchronize before the reduction step.

### Key Optimizations

| Optimization | Description |
|---|---|
| **Pinned Memory** | Host arrays (`u`, `uhelp`, `uvis`, `diffs`) allocated with `cudaMallocHost` — enables fast DMA transfers |
| **Double Buffering** | `swap_mats_dev()` swaps `u_dev` ↔ `uhelp_dev` pointers — O(1) cost instead of O(N²) copy |
| **Memory Coalescing** | Block kernel ensures adjacent threads access contiguous addresses — full warp coalescing |
| **Shared Memory Tiling** | Eliminates redundant global reads; each element loaded once per block per iteration |
| **Intra-block Reduction** | Per-block residual reduction in shared memory reduces global memory writes by 256× |

---

## 🖥️ Multi-GPU Implementation (`PrototipoVariasGPU`)

Built on the best single-GPU strategy (Block + Shared Memory), this prototype scales to **1, 2, or 4 GPUs** using horizontal domain decomposition.

### Domain Decomposition

The grid is split into **horizontal stripes**, one per GPU. For a 2048×2048 grid with 4 GPUs, each GPU owns approximately 512 rows.

### Execution Flow (per iteration)

1. **`cudaSetDevice(i)`** — select each GPU's context.
2. **Jacobi kernel launch** — each GPU computes its assigned stripe.
3. **Halo exchange** — `halo_exchange_multi()` synchronizes border rows between adjacent GPUs using **Peer-to-Peer (P2P)** direct copies (`cudaDeviceEnablePeerAccess`), bypassing host memory entirely.
4. **Residual reduction** — each GPU computes its local error; partial results are combined on the host.

### Controlling Number of GPUs

The number of active GPUs is set via `CUDA_VISIBLE_DEVICES` before launching:

```bash
export CUDA_VISIBLE_DEVICES=0          # 1 GPU
export CUDA_VISIBLE_DEVICES=0,1        # 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 4 GPUs
./heat.exe test.dat -s 2046 -n 250000
```

---

## 📄 Input File Format

The solver reads heat source configuration from a `.dat` file:

```
<num_sources>
<posx> <posy> <range> <temp>
...
```

Example (`test.dat`):
```
2
0.0 0.0 2.0 2.5
0.5 1.0 2.0 2.5
```

Each heat source is defined by:

| Field | Description |
|---|---|
| `posx`, `posy` | Normalized position in [0, 1] |
| `range` | Radius of influence |
| `temp` | Temperature contribution to nearby boundary cells |

---

## 🛠️ Requirements

| Requirement | Version / Notes |
|---|---|
| NVIDIA GPU | RTX 3080 (`sm_86`) or RTX 4090 (`sm_89`) |
| CUDA Toolkit | 12.2.2 — path: `/Soft/cuda/12.2.2` |
| NVCC | Included with CUDA Toolkit |
| GNU Make | Any recent version |
| SLURM | For cluster job submission on Boada |
| Linux | Required (`gettimeofday`, POSIX) |

Verify your environment before building:
```bash
export PATH=/Soft/cuda/12.2.2/bin:$PATH
nvidia-smi
nvcc --version
```

---

## 🚀 Build & Run

### Single-GPU (`PrototipoConjuntoConBlock`)

```bash
cd PrototipoConjuntoConBlock
make
./heat.exe test.dat
```

The `test.dat` file defines the heat sources. The `job.sh` script auto-generates one before launching.

### Multi-GPU (`PrototipoVariasGPU`)

```bash
cd PrototipoVariasGPU
make

# Choose number of GPUs via CUDA_VISIBLE_DEVICES:
export CUDA_VISIBLE_DEVICES=0,1,2,3
./heat.exe test.dat -s 2046 -n 250000
```

### Clean

```bash
make clean
# Removes: *.o, heat.exe, *.ppm, submit-heat*
```

---

## 🖥️ SLURM Cluster Execution

Both prototypes are designed for submission to the **Boada cluster** via SLURM.

### Single-GPU (`job.sh` options)

| Option | GPU | QoS | Notes |
|---|---|---|---|
| A | 1× RTX 4090 | `cuda4090` | Highest single-GPU performance |
| B | 4× RTX 3080 | `cuda3080` | Multi-GPU testing |
| C ✅ | 1× RTX 3080 | `cuda3080` | Default |

```bash
sbatch job.sh
```

### Multi-GPU (`job.sh`)

Requests **4× RTX 3080** by default and runs three configurations (1, 2, 4 GPUs) for comparison:

```bash
sbatch job.sh
# Default: CUDA_VISIBLE_DEVICES=0,1,2,3 → 4 GPUs
# Resolution: 2046×2046, Max iterations: 250,000
```

SLURM output files:
- `submit-heat.o<jobid>` / `submit-heat-multi.o<jobid>` — standard output
- `submit-heat.e<jobid>` / `submit-heat-multi.e<jobid>` — standard error

---

## 📊 Benchmark Results

All benchmarks were run on a **2048×2048 grid** with 250,000 max iterations, 2 heat sources (`test.dat`), and 1× NVIDIA RTX 3080.

### Single-GPU — Kernel Strategy Comparison

| # | Work | Pinned | Shared | Streams | Time (s) |
|---|---|---|---|---|---|
| 1 | Column | No | No | No | 186.86 |
| 2 | Column | Yes | No | No | 187.01 |
| 3 | Column | Yes | Yes | No | 273.71 |
| 4 | Column | Yes | No | Yes | 187.21 |
| 5 | Row | No | No | No | 342.98 |
| 6 | Row | Yes | No | No | 342.24 |
| 7 | Row | Yes | Yes | No | 386.07 |
| 8 | Row | Yes | No | Yes | 343.77 |
| **9** | **Block** | **No** | **Yes** | **No** | **51.33** ✅ |
| 10 | Block | Yes | Yes | No | 51.31 |
| 11 | Block | Yes | Yes | Yes | 52.69 |

**Key findings:**
- **Block tiling is ~3.6× faster than Column and ~6.7× faster than Row** — driven by 2D data locality and full memory coalescing.
- Shared memory hurts Row/Column kernels (overhead > savings) but is essential for Block.
- Pinned memory and Streams show negligible impact when only kernel time is measured.

### Multi-GPU Scaling (Block kernel, 1× RTX 3080 each)

| GPUs | Time (s) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 51.57 | 1.00× | 100% |
| 2 | 35.20 | 1.47× | 73.5% |
| 4 | 29.61 | 1.74× | 43.5% |

The system shows **diminishing returns** at 4 GPUs for this problem size, due to halo exchange overhead and synchronization costs between devices.

---

## 📄 Documentation

| File | Description |
|---|---|
| `PracticasTGA.pdf` | Original assignment statement by Agustín Fernández & Daniel Jiménez (DAC, FIB UPC). Covers problem definition, Jacobi and Gauss-Seidel algorithms, and implementation requirements for 1 and multi-GPU. |
| `Memoria_Práctica_TGA.pdf` | Full technical report by David Castro & David Víctor. Covers design decisions, all kernel variants, optimization analysis, benchmark tables and graphs, and multi-GPU scaling results. |

---

## 🧰 Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/CUDA%20C%2FC%2B%2B-86.7%25-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/C-7.2%25-A8B9CC?style=flat-square&logo=c&logoColor=white" />
  <img src="https://img.shields.io/badge/Shell%20%2F%20SLURM-3.1%25-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white" />
  <img src="https://img.shields.io/badge/Makefile-3.0%25-064F8C?style=flat-square" />
</p>

- **CUDA C/C++** — GPU kernels (row, column, block variants with/without shared memory), CUDA streams, parallel reduction, multi-GPU P2P transfers.
- **C** — Host logic: initialization, boundary conditions, heat source setup, PPM image output, timing (`gettimeofday`).
- **Makefile** — Build automation with `nvcc -O3`, targeting `sm_86` (RTX 3080) and `sm_89` (RTX 4090).
- **Shell / SLURM** — Cluster job submission via `sbatch`, GPU selection with `CUDA_VISIBLE_DEVICES`.

---

## 👤 Authors

**David Castro** & **David Víctor** — [@DavidVJ18](https://github.com/DavidVJ18)

Academic project for the **Tarjetas Gráficas y Aceleradores (TGA)** course at **FIB, Universitat Politècnica de Catalunya (UPC)**, Q1 2025-26.  
Professors: Agustín Fernández & Daniel Jiménez — Departament d'Arquitectura de Computadors.

---

> *If you find this project useful or educational, feel free to leave a ⭐ on the repository.*
