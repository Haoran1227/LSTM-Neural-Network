from Layers.Base import base_layer
import numpy as np
import math

class Sigmoid(base_layer):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        # f(x) = 1/ (1+e^-x)
        self.activation = 1 / (1 + np.power(math.e , -1 * input_tensor))
        return self.activation

    def backward(self, error_tensor):
        # f'(x) = f(x)*(1-f(x))
        error_tensor = error_tensor * self.activation * (1 - self.activation)
        return error_tensor