# Gradient-based optimizers ported from Project 1 (GD, SGD, RMSprop, Adam).
# Agreed interface (see CONTRIBUTING.md):
# - Every optimizer has update(layers, layer_grads), using the same list-of-(W, b) structure.

import numpy as np 

class GD:
    """ 
    Update rule for both plain and stochastic gradient descent. 
    For plain -> use one batch with all the data when passed to train. 
    For stochastic -> Define a batch size, small random batches per epoch. 
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
    def __init__(self, gamma=0.01 ,rho= 0.9, eps=1e-8):
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
    def __init(self,):
        pass 