# `KSP.getResidualNorm()`

## Source Mapping

| Layer | Location |
|---|---|
| **Python method** | `petsc4py/src/PETSc/KSP.pyx` → `getResidualNorm()` |
| **Cython call** | `CHKERR( KSPGetResidualNorm(self.ksp, &rnorm) )` |
| **C function** | `KSPGetResidualNorm(KSP ksp, PetscReal *rnorm)` |
| **C header** | [`include/petscksp.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscksp.h) |
| **C implementation** | [`src/ksp/ksp/interface/itfunc.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/interface/itfunc.c) |

---

## Docstring

```python
def getResidualNorm(self) -> float:
    """
    Return the (possibly preconditioned) residual norm from the last
    completed KSP solve.

    After ``ksp.solve(b, x)`` completes, this method retrieves the
    norm of the residual that was monitored internally during iteration.
    Depending on the norm type (see ``KSP.setNormType``), this is one of:

    - **Preconditioned norm** (default for CG):
        ||B r_k||_2  where r_k = b - A x_k and B is the preconditioner.
    - **Unpreconditioned norm** (default for GMRES):
        ||r_k||_2 = ||b - A x_k||_2, the true residual 2-norm.

    The convergence test compares this norm to the tolerances set by
    ``ksp.setTolerances()``. Specifically, the solver converges when:

        ||r_k|| < max(rtol * ||r_0||, atol)

    where rtol is the relative tolerance and atol is the absolute
    tolerance.

    Wraps the C function ``KSPGetResidualNorm`` declared in
    ``include/petscksp.h`` and implemented in
    ``src/ksp/ksp/interface/itfunc.c``. The C signature is:

        PetscErrorCode KSPGetResidualNorm(KSP ksp, PetscReal *rnorm)

    where ``PetscReal`` maps to Python ``float`` (typically 64-bit
    double precision).

    Parameters
    ----------
    None

    Returns
    -------
    rnorm : float
        The residual norm (>=0) as computed by the KSP solver at the
        final iteration. The precise quantity returned depends on the
        KSP type and norm type:

        - For ``'cg'``: the preconditioned residual norm ||B r_k||_2.
        - For ``'gmres'``: the unpreconditioned residual norm ||r_k||_2.
        - For some methods, if no norm was computed on the final step,
          the norm from the previous iteration is returned.

    Notes
    -----
    This method must be called **after** ``ksp.solve()``. Calling it
    before a solve returns an undefined value (typically 0.0).

    For ``'gmres'``, the residual norm is not computed by an explicit
    inner product but is estimated from the Arnoldi decomposition. It
    equals the true residual norm but is obtained more cheaply.

    To obtain the true unpreconditioned residual norm regardless of
    solver type, compute it explicitly:

    >>> r = b.duplicate()
    >>> A.mult(x, r)         # r = A x
    >>> r.aypx(-1.0, b)      # r = b - A x
    >>> true_norm = r.norm() # ||b - Ax||_2

    Raises
    ------
    petsc4py.PETSc.Error
        If the KSP object has not been set up (i.e., ``solve()`` was
        never called).

    Examples
    --------
    Basic usage — checking solver convergence quality:

    >>> import petsc4py, sys
    >>> petsc4py.init(sys.argv)
    >>> from petsc4py import PETSc
    >>>
    >>> # (build A, b as before)
    >>> ksp = PETSc.KSP().create()
    >>> ksp.setOperators(A)
    >>> ksp.setType('cg')
    >>> ksp.setTolerances(rtol=1e-8, atol=1e-12)
    >>>
    >>> pc = ksp.getPC()
    >>> pc.setType('icc')
    >>>
    >>> x = b.duplicate()
    >>> ksp.solve(b, x)
    >>>
    >>> rnorm = ksp.getResidualNorm()
    >>> iters = ksp.getIterationNumber()
    >>> reason = ksp.getConvergedReason()
    >>>
    >>> print(f"Converged in {iters} iterations")
    >>> print(f"Final residual norm: {rnorm:.4e}")
    >>> print(f"Converged reason code: {reason}")  # positive = converged

    Convergence study — comparing norm vs. iteration count across grids:

    >>> for n in [16, 32, 64, 128]:
    ...     # build and solve n×n Poisson system
    ...     solver = PoissonSolver(n=n)
    ...     u, info = solver.solve()
    ...     print(f"n={n:4d}  iters={info['iterations']:4d}"
    ...           f"  ||r||={info['residual_norm']:.3e}")

    See Also
    --------
    KSP.getIterationNumber : Number of iterations taken in the last solve.
    KSP.getConvergedReason : Integer code indicating why the solver stopped.
    KSP.setTolerances : Set rtol, atol, divergence tolerance, and max iterations.
    KSP.setNormType : Control which norm is used in the convergence test.

    References
    ----------
    PETSc manual page:
        https://petsc.org/release/manualpages/KSP/KSPGetResidualNorm/
    C header declaration:
        https://gitlab.com/petsc/petsc/-/blob/main/include/petscksp.h
    C source (itfunc.c):
        https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/interface/itfunc.c
    """
```
