# `PC.setType(pc_type)`

## Source Mapping

| Layer | Location |
|---|---|
| **Python method** | `petsc4py/src/PETSc/PC.pyx` → `setType()` |
| **Cython call** | `CHKERR( PCSetType(self.pc, cval) )` |
| **C function** | `PCSetType(PC pc, PCType type)` |
| **C header** | [`include/petscpc.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscpc.h) |
| **C implementation** | [`src/ksp/pc/interface/precon.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/pc/interface/precon.c) |

---

## Docstring

```python
def setType(self, pc_type: str) -> None:
    """
    Set the preconditioner algorithm applied within a KSP solve.

    A preconditioner B approximates A^{-1} and transforms the linear
    system Ax = b into the equivalent (but better-conditioned) system

        B A x = B b   (left preconditioning)

    so that the Krylov solver converges in fewer iterations. The
    condition number of BA is generally much smaller than that of A,
    which directly controls convergence speed.

    This method selects which approximation strategy B uses. The best
    choice depends on the structure of A:

    - **ICC** (Incomplete Cholesky): Factorizes A ≈ LL^T, dropping
      fill-in outside a chosen sparsity pattern. Only valid for
      symmetric positive definite (SPD) matrices. Use with ``'cg'``.
    - **ILU** (Incomplete LU): Factorizes A ≈ LU with dropped fill-in.
      Works for general (possibly non-symmetric) sparse matrices.
      Use with ``'gmres'`` or ``'bcgs'``.
    - **Jacobi**: B = diag(A)^{-1}, i.e., diagonal scaling. Cheapest
      option; effective when A is diagonally dominant.
    - **none**: No preconditioning (B = I). Useful for testing or when
      the system is already well-conditioned.

    Wraps the C function ``PCSetType`` declared in ``include/petscpc.h``
    and implemented in ``src/ksp/pc/interface/precon.c``.

    Parameters
    ----------
    pc_type : str
        Name of the preconditioner algorithm. Common values:

        +-------------+---------------------------------------------+
        | String      | Method                                      |
        +=============+=============================================+
        | ``'icc'``   | Incomplete Cholesky (SPD matrices only)     |
        +-------------+---------------------------------------------+
        | ``'ilu'``   | Incomplete LU (general sparse matrices)     |
        +-------------+---------------------------------------------+
        | ``'jacobi'``| Diagonal scaling                            |
        +-------------+---------------------------------------------+
        | ``'sor'``   | Successive Over-Relaxation                  |
        +-------------+---------------------------------------------+
        | ``'bjacobi'``| Block Jacobi (parallel-friendly)           |
        +-------------+---------------------------------------------+
        | ``'hypre'`` | Hypre AMG (requires --download-hypre build) |
        +-------------+---------------------------------------------+
        | ``'none'``  | Identity (no preconditioning)               |
        +-------------+---------------------------------------------+

        The full list is available in ``PCType`` or by running your
        program with ``-help``. Can also be set at runtime via the
        command-line flag ``-pc_type <n>``.

    Returns
    -------
    None

    Notes
    -----
    The PC object is retrieved from a KSP object via ``ksp.getPC()``.
    ``setType()`` must be called before ``ksp.setUp()`` or
    ``ksp.solve()``.

    Mismatching the PC type and the matrix symmetry will either raise
    an error or silently degrade convergence (e.g., using ``'icc'`` on
    a non-SPD matrix will likely fail at factorization time).

    Prefer ``ksp.setFromOptions()`` so the preconditioner can be chosen
    at runtime with ``-pc_type``.

    Raises
    ------
    petsc4py.PETSc.Error
        If ``pc_type`` is not a registered PC type string, or if the
        chosen factorization (e.g., ICC) detects the matrix is not SPD.

    Examples
    --------
    Setting ICC preconditioning for a CG solve on an SPD system:

    >>> import petsc4py, sys
    >>> petsc4py.init(sys.argv)
    >>> from petsc4py import PETSc
    >>>
    >>> A = PETSc.Mat().createAIJ([100, 100], nnz=5)
    >>> # ... (assemble A as SPD, e.g. discrete Laplacian) ...
    >>> A.assemblyBegin(); A.assemblyEnd()
    >>>
    >>> ksp = PETSc.KSP().create()
    >>> ksp.setOperators(A)
    >>> ksp.setType('cg')
    >>>
    >>> pc = ksp.getPC()
    >>> pc.setType('icc')          # Incomplete Cholesky for SPD A
    >>>
    >>> b = A.createVecRight()
    >>> b.set(1.0)
    >>> x = b.duplicate()
    >>> ksp.solve(b, x)

    Using ILU for a non-symmetric system with GMRES:

    >>> ksp.setType('gmres')
    >>> pc = ksp.getPC()
    >>> pc.setType('ilu')          # ILU for general sparse A

    Disabling preconditioning to test raw solver convergence:

    >>> pc.setType('none')

    See Also
    --------
    PC.getType : Retrieve the currently set PC type string.
    KSP.setType : Set the Krylov iterative method.
    KSP.setFromOptions : Override both KSP and PC type from command line.
    KSP.getPC : Retrieve the PC object associated with a KSP.

    References
    ----------
    PETSc manual page:
        https://petsc.org/release/manualpages/PC/PCSetType/
    C source (precon.c):
        https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/pc/interface/precon.c
    """
```
