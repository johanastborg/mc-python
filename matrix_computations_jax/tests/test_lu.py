import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matrix_computations_jax.algorithms.lu import lu_factorization

def test_lu_simple():
    # A = [[1, 2], [3, 4]]
    # No pivoting:
    # multiplier = 3/1 = 3
    # Row2 = Row2 - 3*Row1 = [3, 4] - [3, 6] = [0, -2]
    # U = [[1, 2], [0, -2]]
    # L = [[1, 0], [3, 1]]

    A = jnp.array([[1., 2.], [3., 4.]])
    L, U = lu_factorization(A)

    expected_L = jnp.array([[1., 0.], [3., 1.]])
    expected_U = jnp.array([[1., 2.], [0., -2.]])

    assert jnp.allclose(L, expected_L, atol=1e-5)
    assert jnp.allclose(U, expected_U, atol=1e-5)
    assert jnp.allclose(jnp.dot(L, U), A, atol=1e-5)

def test_lu_random():
    key = jax.random.PRNGKey(42)
    n = 10
    A = jax.random.normal(key, (n, n))

    # Ensure non-singular / no zero pivots by adding diagonal
    A = A + n * jnp.eye(n)

    L, U = lu_factorization(A)

    # Check reconstruction
    assert jnp.allclose(jnp.dot(L, U), A, atol=1e-4)
    # Check L is unit lower triangular
    assert jnp.allclose(L, jnp.tril(L), atol=1e-6)
    assert jnp.allclose(jnp.diag(L), jnp.ones(n), atol=1e-6)
    # Check U is upper triangular
    assert jnp.allclose(U, jnp.triu(U), atol=1e-6)

if __name__ == "__main__":
    test_lu_simple()
    test_lu_random()
    print("LU tests passed!")
