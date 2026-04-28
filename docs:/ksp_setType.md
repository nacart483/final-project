# `KSP.setType(ksp_type)`

## Source Mapping

| Layer | Location |
|---|---|
| **Python method** | `petsc4py/src/PETSc/KSP.pyx` → `setType()` |
| **Cython call** | `CHKERR( KSPSetType(self.ksp, cval) )` |
| **C function** | `KSPSetType(KSP ksp, KSPType type)` |
| **C header** | [`include/petscksp.h`](https://gitlab.com/petsc/petsc/-/blob/main/include/petscksp.h) |
| **C implementation** | [`src/ksp/ksp/interface/itcreate.c`](https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/interface/itcreate.c) |

---

## Docstring

```python
def setType(self, ksp_type: str) -> None:
    """
    Set the Krylov subspace method used to solve the linear system Ax = b.

    Selects which iterative algorithm the KSP object will use when
    ``solve()`` is called. The choice of method should reflect the
    mathematical properties of the operator A:

    - **Symmetric positive definite (SPD) systems** (e.g., the discrete
      Laplacian): use ``'cg'`` (Conjugate Gradient). CG minimizes the
      A-norm of the error over successive Krylov subspaces
      K_k = span{r_0, Ar_0, ..., A^{k-1}r_0} and is optimal for SPD A.
    - **Non-symmetric or indefinite systems**: use ``'gmres'``
      (Generalized Minimal RESidual). GMRES minimizes the 2-norm of the
      residual over K_k and is more robust but requires storing k basis
      vectors.
    - **General non-symmetric systems (low memory)**: use ``'bicg'`` or
      ``'bcgs'`` (BiConjugate Gradient Stabilized).

    Wraps the C function ``KSPSetType`` declared in
    ``include/petscksp.h`` and implemented in
    ``src/ksp/ksp/interface/itcreate.c``.

    Parameters
    ----------
    ksp_type : str
        Name of the Krylov method. Common values:

        +------------+--------------------------------------------+
        | String     | Method                                     |
        +============+============================================+
        | ``'cg'``   | Conjugate Gradient (SPD systems only)      |
        +------------+--------------------------------------------+
        | ``'gmres'``| Generalized Minimal Residual               |
        +------------+--------------------------------------------+
        | ``'bicg'`` | BiConjugate Gradient                       |
        +------------+--------------------------------------------+
        | ``'bcgs'`` | BiCG Stabilized                            |
        +------------+--------------------------------------------+
        | ``'minres'``| Minimal Residual (symmetric indefinite)   |
        +------------+--------------------------------------------+
        | ``'preonly'``| Apply preconditioner once (direct solvers)|
        +------------+--------------------------------------------+

        The full list is available in ``KSPType`` or by running your
        program with ``-help``. Can also be set at runtime via the
        command-line flag ``-ksp_type <name>``.

    Returns
    -------
    None

    Notes
    -----
    Prefer ``ksp.setFromOptions()`` over this routine when possible,
    as the options database lets you switch methods without recompiling.
    Use ``setType()`` only when the solver type must be chosen
    programmatically (e.g., it changes during execution).

    This method must be called before ``ksp.setUp()`` or
    ``ksp.solve()``.

    Raises
    ------
    petsc4py.PETSc.Error
        If ``ksp_type`` is not a registered KSP type string.

    Examples
    --------
    Solving a symmetric positive definite system with CG:

    >>> import petsc4py, sys
    >>> petsc4py.init(sys.argv)
    >>> from petsc4py import PETSc
    >>>
    >>> A = PETSc.Mat().createAIJ([100, 100], nnz=3)
    >>> # ... (assemble A as SPD matrix) ...
    >>> A.assemblyBegin(); A.assemblyEnd()
    >>>
    >>> ksp = PETSc.KSP().create()
    >>> ksp.setOperators(A)
    >>> ksp.setType('cg')          # Conjugate Gradient for SPD A
    >>>
    >>> pc = ksp.getPC()
    >>> pc.setType('icc')          # Incomplete Cholesky preconditioner
    >>>
    >>> b = A.createVecRight()
    >>> b.set(1.0)
    >>> x = b.duplicate()
    >>> ksp.solve(b, x)
    >>> print(ksp.getIterationNumber(), "iterations")

    Switching to GMRES for a non-symmetric problem:

    >>> ksp.setType('gmres')       # works for non-symmetric A
    >>> ksp.setType(PETSc.KSP.Type.GMRES)  # equivalent using enum

    See Also
    --------
    KSP.getType : Retrieve the currently set KSP type string.
    KSP.setFromOptions : Set KSP options from the command-line database.
    PC.setType : Set the preconditioner type for this KSP.
    KSP.setTolerances : Set convergence tolerances.

    References
    ----------
    PETSc manual page:
        https://petsc.org/release/manualpages/KSP/KSPSetType/
    C source (itcreate.c):
        https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/interface/itcreate.c
    """
```
