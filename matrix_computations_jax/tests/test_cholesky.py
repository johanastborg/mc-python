import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matrix_computations_jax.algorithms.cholesky import cholesky_factorization

def test_cholesky_simple():
    # Construct a simple SPD matrix
    # A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    # L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    # A = L @ L.T

    L_true = jnp.array([
        [2., 0., 0.],
        [6., 1., 0.],
        [-8., 5., 3.]
    ])
    A = jnp.dot(L_true, L_true.T)

    G = cholesky_factorization(A)

    assert jnp.allclose(G, L_true, atol=1e-5)
    assert jnp.allclose(jnp.dot(G, G.T), A, atol=1e-5)

def test_cholesky_random_spd():
    key = jax.random.PRNGKey(0)
    n = 10
    # Generate random matrix
    X = jax.random.normal(key, (n, n))
    # Make it SPD
    A = jnp.dot(X, X.T) + 0.1 * jnp.eye(n)

    G = cholesky_factorization(A)

    # Check reconstruction
    assert jnp.allclose(jnp.dot(G, G.T), A, atol=1e-4)
    # Check lower triangular
    assert jnp.allclose(G, jnp.tril(G), atol=1e-6)

if __name__ == "__main__":
    test_cholesky_simple()
    test_cholesky_random_spd()
    print("Cholesky tests passed!")
