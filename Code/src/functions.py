# Cost functions, activation functions and their derivatives (part a).
import jax.numpy as jnp  

def mse(predict, target):
    """Mean squared error between predictions and targets."""
    return jnp.mean((predict - target)**2)

def mse_der(predict, target):
    """Derivative of the MSE with respect to prediction."""
    return 2.0*(predict - target)/predict.size

def L1(W):
    """The L1 regularization penalty as the sum of absolute weights."""
    return jnp.sum(jnp.abs(W))

def L1_der(W):
    """The subgradient of the L1 penalty with respect to the weights."""
    return jnp.sign(W)

def L2(W):
    """The L2 regularization penalty as the sum of squared weights."""
    return jnp.sum(W**2)

def L2_der(W):
    """The derivative of the L2 penalty with respect to the weights."""
    return 2 * W

def cross_entropy(predict, target):
    """Mean multiclass cross-entropy per sample"""
    predict = jnp.clip(predict, 1e-12, 1.0) #clipping to prevent log0
    return jnp.sum(-target * jnp.log(predict)) / predict.shape[0]

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def softmax_cross_entropy_der(predict, target):
    """Gradient of softmax cross-entropy with respect to the logits."""
    return (predict - target) / predict.shape[0]

def sigmoid(z):
    """Sigmoid activation function. Mapping inputs to values between 0 and 1 ."""
    return 1 / (1 + jnp.exp(-z))

def sigmoid_der(z):
    """ Derivative of the sigmoid function with respect to z . """
    s = sigmoid(z)
    return (s)*(1- s)

def ReLU(z):
    """ Relu activation function. Returning max(0, z) elementwise. """
    return jnp.maximum(0.0, z)

def ReLU_der(z):
    """ Derivative of ReLU, returning 1 for positive inputs and 0 otherwise. """
    return jnp.where(z > 0, 1.0, 0.0)

def leaky_ReLU(z, a=0.01):
    """Leaky ReLU, using slope a for negative inputs and slope 1 for positive inputs."""
    return jnp.where(z > 0, z, a * z)

def leaky_ReLU_der(z, a=0.01):
    """ Derivative of Leaky ReLU, returning a for negative inputs and 1 for positive inputs."""
    return jnp.where(z > 0, 1.0, a)

def identity(z):
    """Linear activation: f(z) = z"""
    return z

def identity_der(z):
    """Derivative of linear activation: f'(z) = 1"""
    return jnp.ones_like(z)


