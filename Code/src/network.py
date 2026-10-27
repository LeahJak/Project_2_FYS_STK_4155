# Feed-forward neural network with backpropagation.

# Agreed interface (see CONTRIBUTING.md):
# - Parameters are a list of (W, b), W has shape (n_inputs, n_outputs).
# - Forward pass per layer: a @ W + b.
# - backprop returns gradients in the same structure as the parameters.
import numpy as np 
from src.data import make_batches
from src.functions import L1, L1_der, L2, L2_der 

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        optimizer,
        regularization=None, 
        lmbda = 0.0,
        seed = 2026,
        penalize_bias=True):

        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.regularization = regularization 
        self.lmbda = lmbda 
        self.rng = np.random.default_rng(seed)
        self.layers = self.create_layers(network_input_size, layer_output_sizes)
        self.optimizer = optimizer
        self.penalize_bias = penalize_bias

    def create_layers(self, network_input_size, layer_output_sizes):
        layers = [] 
        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = self.rng.normal(0.0, 1.0 / np.sqrt(i_size), size=(i_size, layer_output_size))
            b = np.zeros(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def predict(self, inputs):
        a = inputs 
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a 

    def penalty(self, layers = None):
        layers = self.layers if layers is None else layers
        terms = [W for W, b in layers] + ([b for W, b in layers] if self.penalize_bias else [])
        if self.regularization == 'L1':
            return self.lmbda * sum(L1(t) for t in terms)
        if self.regularization == 'L2':
                return self.lmbda * sum(L2(t) for t in terms)
        
    def cost(self, inputs, targets, include_penalty = False):
        predict = self.predict(inputs)
        c = self.cost_fun(predict, targets)
        return c + self.penalty() if include_penalty else c

    def jax_compliant_predict(self, layers, inputs):
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            a = activation_func(a @ W + b)
        return a

    def jax_cost(self, layers, inputs, targets):
        pred = self.jax_compliant_predict(layers, inputs)
        return self.cost_fun(pred, targets) + self.penalty(layers)

    def feed_forward_saver(self, inputs):
        layer_inputs, zs , activations= [], [], []
        a = inputs 
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
            activations.append(a)

        return layer_inputs, zs, activations

    def compute_gradient(self, inputs, targets):
        layer_inputs, zs, activations = self.feed_forward_saver(inputs)
        layers = self.layers 
        layer_grads = [() for layer in layers]
        for i in reversed(range(len(layers))):
            layer_input, z = layer_inputs[i], zs[i]
            activation_der = self.activation_ders[i]
            if i == len(layers) - 1:
                dC_da = self.cost_der(activations[-1], targets) #feed_forward_saver returns the activations as a list, the final one is the output activation
            else:
                (W, b) = layers[i + 1]
                dC_da = dC_dz @ W.T                          # (n, M_{l+1}) @ (M_{l+1}, M_l)
                
            dC_dz = dC_da * activation_der(z)                # (n, M_l)
            dC_dW = layer_input.T @ dC_dz                    # (M_{l-1}, n) @ (n, M_l): summed over the samples
            dC_db = np.sum(dC_dz, axis=0)  

            W, b = layers[i] 
            if self.regularization == 'L1':  
                dC_dW += self.lmbda * L1_der(W) 
                if self.penalize_bias:
                    dC_db += self.lmbda * L1_der(b)
            elif self.regularization == 'L2' :        
                dC_dW +=  self.lmbda * L2_der(W)
                if self.penalize_bias:
                    dC_db += self.lmbda * L2_der(b)  
                     
            layer_grads[i] = (dC_dW, dC_db)                  # (M_l,)
        return layer_grads


    def update_weights(self, layer_grads):
        self.layers = self.optimizer.update(self.layers, layer_grads) #each optimizer has(needs per now) its own update method 

    def train(self, inputs, targets, epochs, batch_size, rng):
        n = len(inputs)
        history = []        #saving the cost per epoch to evaluate the learning 
        for epoch in range(epochs):
            batches = make_batches(n, batch_size, rng)

            for batch in batches:
                inputs_batch = inputs[batch]
                targets_batch = targets[batch]
                layer_grads = self.compute_gradient(inputs_batch, targets_batch)
                self.update_weights(layer_grads)
            history.append(self.cost(inputs, targets))
        return history 
