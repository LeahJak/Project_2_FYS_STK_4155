import jax 
import jax.numpy as jnp 
import pytest 

from src.functions import (
    mse,
    mse_der,
    cross_entropy,
    softmax,
    softmax_cross_entropy_der,
    sigmoid,
    sigmoid_der,
    ReLU,
    ReLU_der,
    leaky_ReLU,
    leaky_ReLU_der,
    identity,
    identity_der)

def helper_check_derivative(f, f_der, z):
    """Compare an analytical derivative with JAX's automatic derivative."""
    expected = jax.vmap(jax.grad(f))(z)
    return jnp.allclose(f_der(z), expected, atol=1e-6)

def test_mse_der():
    predict = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([1.0, 3.0, 2.0])

    assert jnp.allclose(
        mse_der(predict, target), jax.grad(lambda p: mse(p, target))(predict))

def test_cross_entropy_der():
    predict = jnp.array([[0.8, 0.2]])
    target = jnp.array([[1.0, 0.0]])

    expected = jax.grad(
        lambda p: cross_entropy(p, target))(predict)

    assert jnp.allclose(
        jax.grad(lambda p: cross_entropy(p, target))(predict), expected)

def test_softmax_cross_entropy_der():
    z = jnp.array([[1.0, 2.0, 3.0]])
    target = jnp.array([[0.0, 0.0, 1.0]])

    expected = jax.grad(lambda z: cross_entropy(softmax(z), target))(z)

    actual = softmax_cross_entropy_der(softmax(z), target)

    assert jnp.allclose(actual, expected, atol=1e-6)

@pytest.mark.parametrize(
    "f, f_der",
    [
        (sigmoid, sigmoid_der),
        (ReLU, ReLU_der),
        (leaky_ReLU, leaky_ReLU_der),
        (identity, identity_der),
    ],)
def test_activation_derivatives(f, f_der):
    z = jnp.array([-2.0, -0.5, 0.5, 2.0])

    assert helper_check_derivative(f, f_der, z)
