# GPU-Accelerated Seam Carving (C + CUDA)

Simple project for content aware image resizing using seam carving.

- GPU does: energy + dynamic programming (when running in GPU mode)
- CPU does: seam backtracking + seam removal
- Saves debug images while processing

## What This Project Generates

After running, you will get:

- `output/seam/seam_*.ppm` : seam highlighted in red (before removal)
- `output/step/step_*.ppm` : image after seam removal
- `output/energy/energy_*.ppm` : energy map snapshots (every 5 iterations)
- `output/energy/energy_final.ppm` : final energy map
- `benchmarks/results.txt` : benchmark summary

Note: these generated outputs are ignored by git (see `.gitignore`).

## Folder and File Uses

### Main folders

- `src/` : C source code (pipeline, image I/O, CPU seam logic, utilities)
- `cuda/` : CUDA kernels and CUDA helpers
- `include/` : header files
- `data/` : input assets (`input.png` is source, `input.ppm` is generated)
- `output/` : generated seam/step/energy images
- `benchmarks/` : generated timing results
- `generate_image/` : helper script to convert `input.png` to `input.ppm`

### Important source files

- `src/main.c` : main flow, CLI parsing, iteration loop
- `src/image.c` : load/save PPM and image rotation helpers
- `src/seam_cpu.c` : CPU energy/DP baseline + seam operations
- `src/utils.c` : timing, benchmark write, energy map normalization/save
- `cuda/energy_kernel.cu` : Sobel energy kernel (shared memory)
- `cuda/dp_kernel.cu` : dynamic programming kernel
- `cuda/cuda_utils.cu` : CUDA memory/copy/error wrappers

## How to Run (Using run.txt)

Use the commands listed in `run.txt` in this order:

1. Clean old build files
2. Generate `data/input.ppm` from `data/input.png` (optional step)
3. Compile CUDA files
4. Compile C files
5. Link to create `seamcarve.exe`
6. Run seam carving
7. View output images

Current run command in `run.txt`:

```bat
seamcarve.exe data/input.ppm output/final.ppm 50 --visualize-energy
```

## Useful Options

Program format:

```bat
seamcarve.exe input.ppm output.ppm <num_seams> [--mode cpu|gpu] [--direction vertical|horizontal] [--interactive] [--visualize-energy]
```

- `--mode cpu|gpu` : choose compute mode (default `gpu`)
- `--direction vertical|horizontal` : seam direction (default `vertical`)
- `--interactive` : pause after each iteration
- `--visualize-energy` : save energy images

## Quick Clean Output (Before Re-run)

```bat
del /Q output\*.ppm output\step\*.ppm output\seam\*.ppm output\energy\*.ppm
```

## Requirements

- Windows build tools (`cl`)
- CUDA Toolkit (`nvcc`)
- Python only if using helper scripts (`generate_image` or `view`)

No external image libraries are used in the C/CUDA pipeline.

## Git Ignore Notes

The repository ignores local/generated artifacts so commits stay clean.

- Build artifacts: `*.obj`, `*.o`, `*.exe`, debug/link logs
- Generated images: `output/*.ppm`, `output/step/*.ppm`, `output/seam/*.ppm`, `output/energy/*.ppm`
- Generated benchmark file: `benchmarks/results.txt`
- Generated converted input: `data/input.ppm`
- Editor/system files: `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`

This means source files are versioned, while run outputs and binaries are not.
