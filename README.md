# Matrix Computations in Python with JAX

This repository contains implementations of numerical linear algebra algorithms from the classic text **"Matrix Computations"** by Gene H. Golub and Charles F. Van Loan. The implementations use **JAX** to leverage hardware acceleration (TPU/GPU) and automatic differentiation capabilities.

## Project Structure

```
matrix_computations_jax/
├── algorithms/       # Core algorithm implementations
│   ├── cholesky.py   # Cholesky factorization
│   ├── lu.py         # LU factorization
│   └── ...
├── utils/            # Utilities for TPU/Device management
├── tests/            # Unit tests
└── examples/         # Usage examples and benchmarks
```

## Prerequisites

- Python 3.9+
- JAX
- Numpy
- Scipy (for comparison/validation)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/matrix_computations_jax.git
   cd matrix_computations_jax
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For TPU support, you must install the TPU-specific version of JAX. See [JAX Installation Guide](https://github.com/google/jax#installation).*
   ```bash
   pip install "jax[tpu]>=0.4.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
   ```

## Usage

### Cholesky Factorization

Algorithm 4.2.1 (Gaxpy Cholesky) implementation.

```python
import jax.numpy as jnp
from matrix_computations_jax.algorithms.cholesky import cholesky_factorization

# Create a Symmetric Positive Definite matrix
A = jnp.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])

# Compute Cholesky factor G such that A = G @ G.T
G = cholesky_factorization(A)
print(G)
```

### LU Factorization

Algorithm 3.2.1 (Gaussian Elimination without Pivoting) implementation.

```python
import jax.numpy as jnp
from matrix_computations_jax.algorithms.lu import lu_factorization

A = jnp.array([[1., 2.], [3., 4.]])

# Compute L and U such that A = L @ U
L, U = lu_factorization(A)
print("L:", L)
print("U:", U)
```

## TPU Optimization Notes

The algorithms in this library are implemented using JAX primitives.

- **JIT Compilation**: All algorithms are decorated with `@jax.jit` to compile them via XLA.
- **Loop Unrolling**: Currently, algorithms use Python loop unrolling for iterations over matrix dimensions. This allows JAX to fully optimize the computational graph but increases compilation time for very large matrices. This approach is chosen to keep the implementation close to the textbook descriptions while still running efficiently on accelerators for moderate sizes.
- **In-Place Updates**: JAX arrays are immutable. The implementations use `jax.numpy` functional updates (e.g., `x.at[idx].set(y)`) which XLA optimizes into in-place mutations on the device memory where possible.

## Algorithms Implemented

| Algorithm | Description | GVL Section | Status |
|-----------|-------------|-------------|--------|
| LU Factorization | Gaussian Elimination without Pivoting | 3.2.1 | ✅ |
| Cholesky | Cholesky Factorization (Gaxpy) | 4.2.1 | ✅ |

## Development

Run tests using pytest:

```bash
pytest matrix_computations_jax/tests/
```

## References

*   Golub, Gene H., and Charles F. Van Loan. *Matrix Computations*. 4th ed. Johns Hopkins University Press, 2013.
*   [JAX Documentation](https://jax.readthedocs.io/)
