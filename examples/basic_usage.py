import jax
import jax.numpy as jnp
from matrix_computations_jax.algorithms.cholesky import cholesky_factorization
from matrix_computations_jax.algorithms.lu import lu_factorization
from matrix_computations_jax.utils.tpu_utils import check_tpu_available

def main():
    print("Checking TPU availability...")
    has_tpu = check_tpu_available()
    print(f"TPU Available: {has_tpu}")

    # 1. Cholesky Example
    print("\n--- Cholesky Factorization Example ---")
    key = jax.random.PRNGKey(0)
    n = 5
    X = jax.random.normal(key, (n, n))
    A = jnp.dot(X, X.T) + 1.0 * jnp.eye(n)

    print("Matrix A (first 2x2):")
    print(A[:2, :2])

    print("Computing Cholesky...")
    G = cholesky_factorization(A)

    print("Factor G (first 2x2):")
    print(G[:2, :2])

    reconstruction_error = jnp.linalg.norm(A - jnp.dot(G, G.T))
    print(f"Reconstruction Error: {reconstruction_error:.2e}")

    # 2. LU Example
    print("\n--- LU Factorization Example ---")
    A_lu = jax.random.normal(key, (n, n)) + 5.0 * jnp.eye(n)

    print("Matrix A (first 2x2):")
    print(A_lu[:2, :2])

    print("Computing LU...")
    L, U = lu_factorization(A_lu)

    reconstruction_error_lu = jnp.linalg.norm(A_lu - jnp.dot(L, U))
    print(f"Reconstruction Error: {reconstruction_error_lu:.2e}")

if __name__ == "__main__":
    main()
