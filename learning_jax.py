import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import numpy as np
from numpy.testing import assert_almost_equal
import os

# This is in here as an attempt to get more precision out of jax
jax.config.update("jax_enable_x64", True)

x = jnp.linspace(-0.5, 0.5, 100, dtype=jnp.float64)
print (x)

# Jfwd = jacfwd(jnp.tanh)(x)
# Jrev = jacrev(jnp.tanh)(x)
#

Jfwd = jnp.diagonal(jacfwd(jnp.tanh)(x.ravel()))
Jfwd_noravel = jnp.diagonal(jacfwd(jnp.tanh)(x))
Jrev = jnp.diagonal(jacrev(jnp.tanh)(x.ravel()))


@jax.jit
def d_tanh(x):
    """
    Compute the derivative of the hyperbolic tangent function.

    Parameters
    ----------
    x : ndarray
        Array value argument

    Returns
    -------
    ndarray
        Derivative of tanh wrt x.
    """
    # The commented out version protects against overflow but slows
    # the code down.
    # idxs_small = np.where(np.abs(x) < 30)
    # d_dx = np.zeros_like(x)
    # d_dx[idxs_small] = 1 / (np.cosh(x[idxs_small]) ** 2)
    d_dx = 1. / (jnp.cosh(x) ** 2)
    return d_dx

d_tanh(x).block_until_ready()