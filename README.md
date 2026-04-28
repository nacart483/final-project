# MA402 Final Project: 2D Poisson Equation with PETSc KSP

**Course:** MA402 — Department of Mathematics, NC State University, Spring 2026\
**Author:** Nicolette Cartwright\
**Domain:** Applied Mathematics — Krylov Subspace Methods for Sparse Linear Systems

------------------------------------------------------------------------

## Problem Summary

This project solves the **2D Poisson equation** on the unit square $\Omega = (0,1)^2$:

$$-\nabla^2 u = f(x,y), \quad u\big|_{\partial\Omega} = 0$$

with manufactured forcing function $f(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$, so the exact solution $u(x,y) = \sin(\pi x)\sin(\pi y)$ is known. This allows rigorous verification of the numerical solver.

The PDE is discretized with a **5-point finite difference stencil** on a uniform $n \times n$ interior grid (spacing $h = 1/(n+1)$), yielding the sparse $N \times N$ linear system $Au = b$ where $N = n^2$.

The system is solved using **PETSc's KSP (Krylov SubSpace) framework** via `petsc4py`, with Conjugate Gradient (CG) as the iterative solver and Incomplete Cholesky (ICC) as the preconditioner — a natural fit because the discrete Laplacian matrix $A$ is symmetric positive definite (SPD).

**Reference C tutorial:** [`src/ksp/ksp/tutorials/ex29.c`](https://petsc.org/main/src/ksp/ksp/tutorials/ex29.c.html)

------------------------------------------------------------------------

## Repository Structure

```         
.
├── README.md                    # This file
├── tutorial_module.py           # petsc4py solver module
├── tutorial_presentation.ipynb  # Jupyter Notebook with simulation + visualization
└── docs/
    ├── ksp_setType.md           # NumPy-style docstring for KSP.setType
    ├── pc_setType.md            # NumPy-style docstring for PC.setType
    └── ksp_getResidualNorm.md   # NumPy-style docstring for KSP.getResidualNorm
```

------------------------------------------------------------------------

## Documented Functions

Three underdocumented `petsc4py` functions are traced from Python through the Cython bridge to their C implementations:

| Python Method | C Function | C Source File |
|------------------------|------------------------|------------------------|
| `KSP.setType(ksp_type)` | `KSPSetType` | [`src/ksp/ksp/interface/itcreate.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/interface/itcreate.c) |
| `PC.setType(pc_type)` | `PCSetType` | [`src/ksp/pc/interface/precon.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/pc/interface/precon.c) |
| `KSP.getResidualNorm()` | `KSPGetResidualNorm` | [`src/ksp/ksp/interface/itfunc.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/interface/itfunc.c) |

Full NumPy-style docstrings (with parameter tables, math, MWEs, and GitLab links) are in `docs/`.

------------------------------------------------------------------------

## Results

-   **Correctness:** The discrete $L^2$ error converges at $O(h^2)$, consistent with the second-order 5-point stencil.
-   **Preconditioner impact:** ICC reduces iteration count by more than 10× compared to no preconditioning, with no change in solution accuracy.
-   **Scalability:** Iteration count grows sub-linearly with system size $N$ under ICC.

| $n$ |      $h$ | $\|u_h - u\|_{L^2}$ | Convergence rate | KSP iters |
|----:|---------:|--------------------:|-----------------:|----------:|
|   8 | 0.111111 |           \~4.6e-03 |                — |      \~15 |
|  16 | 0.058824 |           \~1.2e-03 |            \~2.0 |      \~25 |
|  32 | 0.030303 |           \~3.0e-04 |            \~2.0 |      \~40 |
|  64 | 0.015385 |           \~7.5e-05 |            \~2.0 |      \~65 |
| 128 | 0.007752 |           \~1.9e-05 |            \~2.0 |     \~110 |

*(Exact values will vary slightly; run the notebook to reproduce.)*

------------------------------------------------------------------------

## AI Translation Experience

### Tools used

The initial `petsc4py` solver script was generated with the help of **Google Gemini** (following the project workflow), then manually debugged and extended.

### What the AI got right

-   The overall structure of the solver class was sound: separate methods for matrix assembly, RHS assembly, solving, and plotting.
-   The KSP setup sequence (`create → setOperators → setType → setTolerances → solve`) was correctly ordered.
-   Basic use of `PETSc.Mat().createAIJ` and `PETSc.Vec().createSeq` was accurate.

### What required debugging

-   **Boundary indexing:** The AI initially included boundary nodes in the unknown vector, leading to an off-by-one error in the matrix size. The fix was to restrict unknowns strictly to the $n$ interior nodes per dimension.
-   **Matrix assembly flags:** The AI omitted `assemblyBegin()` / `assemblyEnd()` calls, which are required before any matrix operations in PETSc — this caused a silent error until explicitly added.
-   **ICC + CG compatibility:** The AI suggested `'ilu'` as the default preconditioner. ILU is designed for non-symmetric systems; for our SPD discrete Laplacian, `'icc'` (Incomplete Cholesky) is the correct and more efficient choice.
-   **`h` definition:** The AI computed grid spacing as $h = 1/n$ (including boundary nodes in the count) rather than the correct $h = 1/(n+1)$ for $n$ interior nodes, leading to a subtle but measurable error in the stencil coefficients.

### Key lesson

AI-generated `petsc4py` code is a useful starting scaffold, but requires careful review of: (1) boundary condition handling, (2) PETSc's required assembly calls, and (3) matching the solver/preconditioner to the mathematical properties of the matrix (SPD vs. non-symmetric).

------------------------------------------------------------------------

## Dependencies

-   [`petsc4py`](https://petsc.org/release/petsc4py/) (with PETSc ≥ 3.20)
-   `numpy`
-   `matplotlib`
-   `jupyter` (for the notebook)

Install via:

``` bash
pip install petsc4py numpy matplotlib jupyter
```

Or, if using a conda environment with PETSc:

``` bash
conda install -c conda-forge petsc4py numpy matplotlib
```

## Running the code

**Solver script only:**

``` bash
python tutorial_module.py
```

**With PETSc command-line overrides** (e.g., switch to GMRES + ILU):

``` bash
python tutorial_module.py -ksp_type gmres -pc_type ilu
```

**Jupyter Notebook:**

``` bash
jupyter notebook tutorial_presentation.ipynb
```

------------------------------------------------------------------------

## References

-   PETSc KSP User Manual: <https://petsc.org/release/manual/ksp/>
-   `KSPSetType` manual page: <https://petsc.org/release/manualpages/KSP/KSPSetType/>
-   `PCSetType` manual page: <https://petsc.org/release/manualpages/PC/PCSetType/>
-   `KSPGetResidualNorm` manual page: <https://petsc.org/release/manualpages/KSP/KSPGetResidualNorm/>
-   PETSc GitLab repository: <https://gitlab.com/petsc/petsc>
