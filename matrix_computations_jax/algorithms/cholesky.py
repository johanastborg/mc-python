import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def cholesky_factorization(A):
    """
    Computes the Cholesky factorization of a symmetric positive definite matrix A.
    A = G @ G.T (Lower triangular G).

    Ref: Golub & Van Loan, Algorithm 4.2.1 (Gaxpy Cholesky) - adapted for JAX

    Args:
        A: Input matrix of shape (n, n).

    Returns:
        G: Lower triangular matrix.
    """
    n = A.shape[0]
    G = jnp.zeros_like(A)

    # We will iterate column by column.
    # We unroll the loop using Python range(n) so that 'j' is a static integer.
    # This allows standard slicing G[j, :j] to work within JIT.

    for j in range(n):
        # v = A[j, j] - G[j, :j] @ G[j, :j].T
        # G[j, :j] is a slice of size (j,). Since j is static, this is fine.
        v = A[j, j] - jnp.dot(G[j, :j], G[j, :j])

        # G[j, j] = sqrt(v)
        scale = jnp.sqrt(v)
        G = G.at[j, j].set(scale)

        # G[j+1:, j] = (A[j+1:, j] - G[j+1:, :j] @ G[j, :j].T) / scale
        # We need to update column j for rows k > j.
        # We can do this in a vectorized way for the column segment below diagonal.

        if j < n - 1:
            # Vectorized update for the column segment
            # G[j+1:, :j] has shape (n-(j+1), j)
            # G[j, :j] has shape (j,)
            # dot product results in shape (n-(j+1),)

            sub = jnp.dot(G[j+1:, :j], G[j, :j])
            val = (A[j+1:, j] - sub) / scale
            G = G.at[j+1:, j].set(val)

    return G
