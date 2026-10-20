# Gradient-based optimizers ported from Project 1 (GD, SGD, RMSprop, Adam).
# Agreed interface (see CONTRIBUTING.md):
# - Every optimizer has update(layers, layer_grads), using the same list-of-(W, b) structure.

import numpy as np 

class GD:
    """ 
    Update rule for both plain and stochastic gradient descent. 
    For plain -> use one batch with all the data when passed to train. 
    For stochastic -> Define a batch size, small random batches per epoch. 

    Parameters:
    gamma: Learning rate. 
    """
    def __init__(self, gamma=0.01):
        self.gamma = gamma 

    def update(self, layers, layer_grads):
        new_layers = [] 
        for (W, b), (dW, db) in zip(layers, layer_grads):
            W = W - self.gamma * dW
            b = b - self.gamma * db
            new_layers.append((W, b))
        return new_layers

class RMSprop:
    """
    Uses an exponential moving average of squared gradients to adapt
    the learning rate for each weight and bias.

    Parameters
    gamma : Learning rate.
    rho : Decay rate for the moving average of squared gradients.
    eps : Small constant for numerical stability.
    """
    def __init__(self, gamma=0.01 ,rho=0.9, eps=1e-8):
        self.gamma = gamma 
        self.rho = rho 
        self.eps = eps 
        self.s = None 

    def update(self, layers, layer_grads):
        if self.s is None:      #only initialise s once when update is called 
            self.s = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]

        new_s = []
        new_layers = [] 
        for (W,b), (dW, db), (s_W, s_b) in zip(layers, layer_grads, self.s ):
            s_W = self.rho * s_W + (1 - self.rho)*dW**2
            s_b = self.rho * s_b + (1- self.rho)*db**2
            new_s.append((s_W, s_b))

            W = W - self.gamma /(np.sqrt(s_W) + self.eps) * dW
            b = b - self.gamma / (np.sqrt(s_b) + self.eps) * db
            new_layers.append((W, b))
        self.s = new_s 
        return new_layers 


class Adam:
    """
    Updates the weights and biases using exponential moving averages
    of the gradients and squared gradients, with bias correction.

    Parameters:
    beta1 : Decay rate for the moving average of the gradients.
    beta2 : Decay rate for the moving average of the squared gradients.
    gamma : Learning rate.
    t : Initial timestep.
    eps : Small constant for numerical stability.
    """
    def __init__(self, gamma=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.gamma = gamma 
        self.beta1 = beta1 
        self.beta2 = beta2
        self.eps = eps 
        self.m = None 
        self.v = None 
        self.t = 0


    def update(self, layers, layer_grads):
        if self.m is None:
            self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
            self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]

        self.t += 1
        new_m = []
        new_v = [] 
        new_layers = []
        for (W, b), (dW, db), (m_W, m_b), (v_W, v_b) in zip(layers, layer_grads, self.m, self.v):
            m_W = self.beta1 * m_W + (1 - self.beta1) * dW
            m_b = self.beta1 * m_b + (1 - self.beta1) * db
            v_W = self.beta2 * v_W + (1 - self.beta2) * dW**2
            v_b = self.beta2 * v_b + (1 - self.beta2) * db**2
            new_m.append((m_W, m_b))
            new_v.append((v_W, v_b))

            m_hat_W = m_W / (1 - self.beta1**self.t)
            m_hat_b = m_b / (1 - self.beta1**self.t)
            v_hat_W = v_W / (1 - self.beta2**self.t)
            v_hat_b = v_b / (1 - self.beta2**self.t)

            W = W - self.gamma * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            b = b - self.gamma * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
            new_layers.append((W, b))
        self.m = new_m
        self.v = new_v 
        return new_layers

