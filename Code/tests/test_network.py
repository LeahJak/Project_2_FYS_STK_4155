import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression 

from src.network import NeuralNetwork
from src.functions import *
from src.optimizers import GD 


class DummyOptimizer:
    def update(self, layers, layer_grads):
        return layers

def jax_cost(layers, X, Y, activations, cost):
    """Forward pass and cost independant of NeuralNetwork."""
    a = X
    for (W, b), f in zip(layers, activations):
        a = f(a @ W + b)
    return cost(a, Y)

jax_mse = lambda a, y: jnp.mean((a - y) ** 2)
jax_ce = lambda a, y: -jnp.sum(y * jnp.log(a)) / a.shape[0]

rng = np.random.default_rng(0)
X_reg, Y_reg = rng.normal(size=(10, 2)), rng.normal(size=(10, 1))
X_clf = rng.normal(size=(10, 4))
Y_clf = np.eye(3)[rng.integers(0, 3, 10)]

CASES = {
    "sigmoid_mse": (X_reg, Y_reg, [5, 4, 1],
        [sigmoid, sigmoid, identity], [sigmoid_der, sigmoid_der, identity_der],
        mse, mse_der,
        [jax.nn.sigmoid, jax.nn.sigmoid, lambda z: z], jax_mse),
    "relu_mse": (X_reg, Y_reg, [5, 4, 1],
        [ReLU, leaky_ReLU, identity], [ReLU_der, leaky_ReLU_der, identity_der],
        mse, mse_der,
        [jax.nn.relu, lambda z: jax.nn.leaky_relu(z, 0.01), lambda z: z], jax_mse),
    "softmax_ce": (X_clf, Y_clf, [5, 4, 3],
        [sigmoid, sigmoid, softmax], [sigmoid_der, sigmoid_der, identity_der],
        cross_entropy, softmax_cross_entropy_der,
        [jax.nn.sigmoid, jax.nn.sigmoid, lambda z: jax.nn.softmax(z, axis=1)], jax_ce),
}

@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_gradient_matches_jax(case):
    """
    Compare backpropagation gradients with jax.grad for several
    activation/cost pairings, including softmax with cross entropy.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, October 2026)
    Role: Suggested the parametrized structure and the test cases,
    based on our original single-case tests format.
    Modifications: added parametrization and dummy optimizer class. 
    """
    X, Y, sizes, acts, ders, cost, cost_der, jax_acts, jax_cost_fn = case
    net = NeuralNetwork(X.shape[1], sizes, acts, ders, cost, cost_der,
                        optimizer=DummyOptimizer(), seed=2026)

    manual = net.compute_gradient(X, Y)
    auto = jax.grad(jax_cost)(net.layers, X, Y, jax_acts, jax_cost_fn)

    assert len(manual) == len(auto) == len(sizes)
    for (mW, mb), (aW, ab) in zip(manual, auto):
        np.testing.assert_allclose(mW, aW, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(mb, ab, rtol=1e-6, atol=1e-10)


def test_network_on_ols():
    # Test data for the ols check 
    rng = np.random.default_rng(2026)
    X = rng.uniform(-1, 1, (100, 1))
    y = 2 * X + 3

    ols = LinearRegression().fit(X, y) 
    y_ols = ols.predict(X)

    network = NeuralNetwork(
        network_input_size=1,
        layer_output_sizes=[1],
        activation_funcs=[identity],
        activation_ders=[identity_der],
        cost_fun=mse,
        cost_der=mse_der,
        optimizer=GD(gamma=0.1),
        seed=2026)
    #full dataset per/batch ran once per epoch -> plain GD. 
    network.train(X, y, epochs=2000, batch_size=len(X), rng=rng) 
    y_network = network.predict(X) 

    (W, b), = network.layers
    np.testing.assert_allclose(W, ols.coef_.T, atol=1e-6)
    np.testing.assert_allclose(b, ols.intercept_, atol=1e-6)
    np.testing.assert_allclose(y_ols, y_network, atol=1e-6)