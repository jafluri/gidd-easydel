import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Set up JAX for testing
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

@pytest.fixture
def key():
    """Provide a consistent random key for tests."""
    return jax.random.PRNGKey(42)

@pytest.fixture
def vocab_size():
    """Common vocabulary size for tests."""
    return 100

@pytest.fixture
def mask_token_id():
    """Common mask token ID for tests."""
    return 99

@pytest.fixture
def batch_shape():
    """Common batch shape for tests."""
    return (4, 8)