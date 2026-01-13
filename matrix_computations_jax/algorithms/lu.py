import jax
import jax.numpy as jnp

@jax.jit
def lu_factorization(A):
    """
    Computes the LU factorization of a matrix A without pivoting.
    A = L @ U

    Ref: Golub & Van Loan, Algorithm 3.2.1 (Gaussian Elimination without Pivoting)

    Args:
        A: Input matrix of shape (n, n).

    Returns:
        L: Lower triangular matrix with unit diagonal.
        U: Upper triangular matrix.
    """
    n = A.shape[0]

    # We will perform the updates in-place on a copy of A,
    # effectively storing L (strictly lower) and U (upper) in the same matrix.

    # Using python loop unrolling for static indices
    for k in range(n-1):
        # Multipliers: A[k+1:, k] = A[k+1:, k] / A[k, k]
        pivot = A[k, k]

        # Calculate multipliers
        # A[k+1:, k] is slice of size n-(k+1)
        multipliers = A[k+1:, k] / pivot

        # Update the column with multipliers (part of L)
        A = A.at[k+1:, k].set(multipliers)

        # Update the rest of the submatrix
        # A[k+1:, k+1:] = A[k+1:, k+1:] - multipliers[:, None] @ A[k, k+1:][None, :]

        sub_update = jnp.outer(multipliers, A[k, k+1:])
        A = A.at[k+1:, k+1:].add(-sub_update)

    LU_packed = A

    # Extract L and U
    L = jnp.tril(LU_packed, -1) + jnp.eye(n)
    U = jnp.triu(LU_packed)

    return L, U
