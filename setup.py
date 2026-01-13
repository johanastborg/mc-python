from setuptools import setup, find_packages

setup(
    name="matrix_computations_jax",
    version="0.1.0",
    description="Implementations of algorithms from Matrix Computations (Golub & Van Loan) using JAX for TPU acceleration.",
    author="Jules",
    packages=find_packages(),
    install_requires=[
        "jax",
        "numpy",
        "scipy",
    ],
    extras_require={
        "test": ["pytest"],
    },
)
