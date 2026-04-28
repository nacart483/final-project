"""
2D Poisson Equation Solver using PETSc KSP (Krylov Subspace Methods)
=====================================================================

This module solves the 2D Poisson equation on the unit square [0,1] x [0,1]:

    -∇²u = f(x, y),    (x, y) ∈ Ω = (0,1)²
        u = 0,          (x, y) ∈ ∂Ω   (Dirichlet boundary conditions)

with the manufactured right-hand side:

    f(x, y) = 2π² sin(πx) sin(πy)

which admits the exact solution:

    u_exact(x, y) = sin(πx) sin(πy)

This allows us to compute the L2 discretization error and verify solver correctness.

The PDE is discretized using a standard 5-point finite difference stencil on a
uniform n×n grid with spacing h = 1/(n+1). The discrete system Au = b is solved
using PETSc's KSP (Krylov SubSpace) framework with a Conjugate Gradient (CG)
solver preconditioned by Incomplete Cholesky (ICC).

Reference C tutorial:
    https://petsc.org/main/src/ksp/ksp/tutorials/ex29.c.html

Dependencies:
    - petsc4py
    - numpy
    - matplotlib
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# --- PETSc Initialization (must happen before any PETSc objects are created) ---
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


class PoissonSolver:
    """
    Solves the 2D Poisson equation -∇²u = f on the unit square using
    PETSc's KSP (Krylov subspace) iterative solvers.

    The domain Ω = (0,1)² is discretized on a uniform (n+2)×(n+2) grid
    (including boundary nodes). Interior nodes form an n×n system. The
    5-point finite difference stencil yields the sparse linear system:

        A u = b

    where A is the standard 2D discrete Laplacian matrix of size n²×n².

    Parameters
    ----------
    n : int
        Number of interior grid points in each dimension. Total interior
        unknowns = n². Grid spacing h = 1/(n+1).
    ksp_type : str
        PETSc KSP solver type string. Default is 'cg' (Conjugate Gradient),
        which is optimal for symmetric positive definite systems like this one.
        Other valid options: 'gmres', 'bicg', 'minres'.
    pc_type : str
        PETSc PC preconditioner type string. Default is 'icc' (Incomplete
        Cholesky). Other valid options: 'ilu', 'jacobi', 'none'.
    rtol : float
        Relative convergence tolerance for the KSP solver. The solver stops
        when ||r_k|| / ||r_0|| < rtol. Default is 1e-8.
    """

    def __init__(self, n=64, ksp_type='cg', pc_type='icc', rtol=1e-8):
        self.n = n
        self.h = 1.0 / (n + 1)          # grid spacing
        self.ksp_type = ksp_type
        self.pc_type = pc_type
        self.rtol = rtol
        self.N = n * n                   # total number of unknowns

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _ij_to_idx(self, i, j):
        """Convert 2D grid index (i, j) to 1D matrix row index."""
        return i * self.n + j

    def _grid_coords(self):
        """
        Return 1D arrays of interior x and y coordinates.

        The interior nodes are at positions h, 2h, ..., n*h
        where h = 1/(n+1).
        """
        coords = np.linspace(self.h, 1.0 - self.h, self.n)
        return coords

    # ------------------------------------------------------------------
    # Phase 1: Assemble the matrix A (discrete Laplacian)
    # ------------------------------------------------------------------

    def assemble_matrix(self):
        """
        Assemble the sparse matrix A for the 5-point finite difference
        discretization of -∇²u.

        The stencil for interior node (i, j) is:

            (4u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}) / h²

        Boundary contributions (where neighbors are on ∂Ω, so u=0 there)
        are simply dropped, which is correct for homogeneous Dirichlet BCs.

        Returns
        -------
        A : PETSc.Mat
            Assembled sparse matrix of size N×N (N = n²), stored in AIJ
            (compressed sparse row) format.
        """
        N = self.N
        h2 = self.h ** 2

        # Pre-allocate: each row has at most 5 nonzeros (diagonal + 4 neighbors)
        A = PETSc.Mat().createAIJ([N, N], nnz=5)
        A.setUp()

        for i in range(self.n):
            for j in range(self.n):
                row = self._ij_to_idx(i, j)

                # Diagonal entry: 4/h²
                A.setValue(row, row, 4.0 / h2)

                # Left neighbor (i, j-1)
                if j > 0:
                    col = self._ij_to_idx(i, j - 1)
                    A.setValue(row, col, -1.0 / h2)

                # Right neighbor (i, j+1)
                if j < self.n - 1:
                    col = self._ij_to_idx(i, j + 1)
                    A.setValue(row, col, -1.0 / h2)

                # Bottom neighbor (i-1, j)
                if i > 0:
                    col = self._ij_to_idx(i - 1, j)
                    A.setValue(row, col, -1.0 / h2)

                # Top neighbor (i+1, j)
                if i < self.n - 1:
                    col = self._ij_to_idx(i + 1, j)
                    A.setValue(row, col, -1.0 / h2)

        # Communicate off-process entries and finalize the matrix
        A.assemblyBegin()
        A.assemblyEnd()
        return A

    # ------------------------------------------------------------------
    # Phase 2: Assemble the right-hand side vector b
    # ------------------------------------------------------------------

    def assemble_rhs(self):
        """
        Assemble the right-hand side vector b = f(x_i, y_j) evaluated at
        each interior grid node.

        The forcing function is:
            f(x, y) = 2π² sin(πx) sin(πy)

        chosen so that the exact solution is u(x, y) = sin(πx) sin(πy).

        Returns
        -------
        b : PETSc.Vec
            RHS vector of length N = n².
        """
        b = PETSc.Vec().createSeq(self.N)
        coords = self._grid_coords()

        b_array = np.zeros(self.N)
        for i, xi in enumerate(coords):
            for j, yj in enumerate(coords):
                row = self._ij_to_idx(i, j)
                b_array[row] = 2.0 * np.pi**2 * np.sin(np.pi * xi) * np.sin(np.pi * yj)

        b.setValues(range(self.N), b_array)
        b.assemblyBegin()
        b.assemblyEnd()
        return b

    # ------------------------------------------------------------------
    # Phase 3: Configure and run the KSP solver
    # ------------------------------------------------------------------

    def solve(self):
        """
        Assemble the linear system and solve it with PETSc KSP.

        Configures a KSP solver with:
            - Solver type:        self.ksp_type  (default: 'cg')
            - Preconditioner:     self.pc_type   (default: 'icc')
            - Relative tolerance: self.rtol      (default: 1e-8)

        Returns
        -------
        u_vec : numpy.ndarray, shape (n, n)
            The numerical solution reshaped onto the interior n×n grid.
        info : dict
            Solver diagnostics with keys:
                'iterations'     : int   — number of KSP iterations taken
                'residual_norm'  : float — final preconditioned residual norm
                'converged_reason': int  — PETSc convergence reason code
                                          (positive = converged, negative = diverged)
        """
        print(f"\n{'='*55}")
        print(f"  Poisson Solver: n={self.n}, N={self.N} unknowns")
        print(f"  KSP type : {self.ksp_type}")
        print(f"  PC type  : {self.pc_type}")
        print(f"  rtol     : {self.rtol}")
        print(f"{'='*55}\n")

        # Assemble system
        A = self.assemble_matrix()
        b = self.assemble_rhs()
        x = b.duplicate()   # solution vector, same size/layout as b

        # --- Configure the KSP solver ---
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)

        # Set the Krylov method (e.g., CG, GMRES, BiCG)
        ksp.setType(self.ksp_type)

        # Configure the preconditioner
        pc = ksp.getPC()
        pc.setType(self.pc_type)

        # Set convergence tolerances:
        #   rtol  — relative tolerance  ||r_k|| / ||r_0|| < rtol
        #   atol  — absolute tolerance  ||r_k|| < atol
        #   dtol  — divergence tolerance (solver gives up if residual grows this much)
        #   max_it — maximum number of iterations
        ksp.setTolerances(rtol=self.rtol, atol=1e-12, divtol=1e5, max_it=10000)

        # Allow command-line overrides (e.g. -ksp_type gmres -pc_type ilu)
        ksp.setFromOptions()

        # --- Solve Au = b ---
        ksp.solve(b, x)

        # --- Retrieve diagnostics ---
        iterations = ksp.getIterationNumber()
        residual_norm = ksp.getResidualNorm()
        converged_reason = ksp.getConvergedReason()

        print(f"  Iterations      : {iterations}")
        print(f"  Residual norm   : {residual_norm:.6e}")
        print(f"  Converged reason: {converged_reason}  "
              f"({'converged' if converged_reason > 0 else 'DIVERGED'})\n")

        # Reshape solution into 2D grid
        u_flat = x.getArray().copy()
        u_vec = u_flat.reshape((self.n, self.n))

        info = {
            'iterations': iterations,
            'residual_norm': residual_norm,
            'converged_reason': converged_reason,
        }
        return u_vec, info

    # ------------------------------------------------------------------
    # Phase 4: Error analysis
    # ------------------------------------------------------------------

    def compute_error(self, u_num):
        """
        Compute the discrete L2 error between the numerical solution and
        the exact solution u_exact(x, y) = sin(πx) sin(πy).

        The discrete L2 norm is defined as:

            ||e||_{L2} = h * sqrt( Σ_{i,j} (u_num[i,j] - u_exact(x_i, y_j))² )

        Parameters
        ----------
        u_num : numpy.ndarray, shape (n, n)
            Numerical solution on the interior grid.

        Returns
        -------
        l2_error : float
            Discrete L2 norm of the error.
        u_exact : numpy.ndarray, shape (n, n)
            Exact solution evaluated on the interior grid.
        """
        coords = self._grid_coords()
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        error = u_num - u_exact
        l2_error = self.h * np.sqrt(np.sum(error**2))
        print(f"  Discrete L2 error: {l2_error:.6e}")
        return l2_error, u_exact

    # ------------------------------------------------------------------
    # Phase 5: Visualization
    # ------------------------------------------------------------------

    def plot_solution(self, u_num, u_exact=None):
        """
        Visualize the numerical solution and (optionally) the exact solution
        and pointwise error as side-by-side contour plots.

        Parameters
        ----------
        u_num : numpy.ndarray, shape (n, n)
            Numerical solution on the interior grid.
        u_exact : numpy.ndarray, shape (n, n), optional
            Exact solution for comparison. If provided, a 3-panel figure is
            shown: numerical solution, exact solution, and pointwise error.
            If None, only the numerical solution is plotted.
        """
        coords = self._grid_coords()
        X, Y = np.meshgrid(coords, coords, indexing='ij')

        if u_exact is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            titles = ['Numerical Solution', 'Exact Solution', 'Pointwise Error']
            data = [u_num, u_exact, np.abs(u_num - u_exact)]
            cmaps = ['viridis', 'viridis', 'magma']
        else:
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            axes = [axes]
            titles = ['Numerical Solution']
            data = [u_num]
            cmaps = ['viridis']

        for ax, title, d, cmap in zip(axes, titles, data, cmaps):
            cf = ax.contourf(X, Y, d, levels=30, cmap=cmap)
            fig.colorbar(cf, ax=ax)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')

        fig.suptitle(
            f'2D Poisson Equation  |  n={self.n}, KSP={self.ksp_type}, PC={self.pc_type}',
            fontsize=13, y=1.02
        )
        plt.tight_layout()
        plt.savefig('poisson_solution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  Figure saved to poisson_solution.png")


# ---------------------------------------------------------------------------
# Execution block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Default run: CG + ICC on a 64x64 grid ---
    solver = PoissonSolver(n=64, ksp_type='cg', pc_type='icc', rtol=1e-8)
    u_num, info = solver.solve()
    l2_error, u_exact = solver.compute_error(u_num)
    solver.plot_solution(u_num, u_exact)

    # --- Optional: convergence study across grid sizes ---
    print("\n--- Convergence Study (CG + ICC) ---")
    print(f"{'n':>6}  {'h':>10}  {'L2 Error':>12}  {'Iters':>6}")
    print("-" * 40)
    for n in [8, 16, 32, 64, 128]:
        s = PoissonSolver(n=n, ksp_type='cg', pc_type='icc', rtol=1e-10)
        u, _ = s.solve()
        err, _ = s.compute_error(u)
        print(f"{n:>6}  {s.h:>10.6f}  {err:>12.4e}")
