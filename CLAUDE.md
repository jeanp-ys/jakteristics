# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

**Development Setup:**
Have `uv` installed.

**Testing:**
```bash
uv run pytest
```

## Architecture Overview

Jakteristics is a Python package for computing geometric features from 3D point clouds. The core architecture consists of:

**Core Components:**
- **Cython Extensions**: Performance-critical code is implemented in Cython (.pyx files)
  - `jakteristics/extension.pyx`: Main feature computation algorithms
  - `jakteristics/utils.pyx`: Utility functions
  - `jakteristics/ckdtree/ckdtree.pyx`: Spatial data structure for neighbor queries

**Key Modules:**
- `main.py`: Primary API functions (`compute_features`, `compute_scalars_stats`)
- `constants.py`: Feature name definitions (28 geometric features available)
- `las_utils.py`: LAS file I/O operations for point cloud data
- `__main__.py`: CLI interface using Typer

**Build System:**
- Uses setuptools with Cython compilation
- Requires NumPy, SciPy, and OpenMP for parallel computation
- Custom BuildExt class handles platform-specific compiler flags
- Extensions are compiled with OpenMP support for multi-threading

**Feature Computation:**
The package computes 28 geometric features based on eigenvalue analysis of local neighborhoods:
- Shape descriptors (planarity, linearity, sphericity)
- Statistical measures (eigenentropy, omnivariance, anisotropy)
- Spatial orientation (verticality, normal vectors)
- Eigenvalues and eigenvectors

**Spatial Queries:**
Uses a custom cKDTree implementation (extended from SciPy) for efficient neighbor searches with radius-based queries.

## Development Notes

- Always run `make build` after modifying .pyx files to compile Cython extensions
- The package supports both Euclidean and Manhattan distance metrics
- Multi-threading is handled via OpenMP with configurable thread counts
- Test data is located in `tests/data/` with LAS format point cloud files