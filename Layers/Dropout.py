import numpy as np
from Layers.Base import base_layer

class Dropout(base_layer):
    def __init__(self, probability):
        super().__init__()
        self.prob = probability         #probability determining the fraction units to keep
        self.dropout_vec = None         #stores dropout vector generated by Bernoulli distribution for backward pass

    def forward(self, input_tensor):
        # in test phase, the dropout has no effect.
        # in training phase, inverted dropout is implemented. 1) multiply dropout vector. 2) rescale by 1/p
        if self.phase == 'train':
            self.dropout_vec = np.random.binomial(1, self.prob, size=input_tensor.shape)
            input_tensor = input_tensor * self.dropout_vec / self.prob
        return input_tensor

    def backward(self, error_tensor):
        # In backward pass, error_tensor multiply the same dropout vector as forward pass
        error_tensor = error_tensor * self.dropout_vec
        return error_tensor

