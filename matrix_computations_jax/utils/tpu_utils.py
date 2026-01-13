import jax
import logging

def check_tpu_available():
    """
    Checks if TPU devices are available for JAX.
    Returns True if TPU is available, False otherwise.
    """
    try:
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == 'tpu']
        if tpu_devices:
            logging.info(f"TPU devices found: {tpu_devices}")
            return True
        else:
            logging.warning("No TPU devices found. JAX will default to CPU or GPU.")
            return False
    except RuntimeError as e:
        logging.error(f"Error checking for devices: {e}")
        return False

def get_device_mesh(mesh_shape):
    """
    Creates a JAX device mesh for parallelism.
    Args:
        mesh_shape: Tuple representing the dimensions of the mesh.
    """
    # This is a placeholder for more advanced TPU mesh configuration
    devices = jax.devices()
    n_devices = len(devices)
    expected_devices = 1
    for dim in mesh_shape:
        expected_devices *= dim

    if n_devices < expected_devices:
        raise ValueError(f"Not enough devices. Requested {expected_devices}, found {n_devices}")

    return jax.sharding.Mesh(devices[:expected_devices], mesh_shape)
