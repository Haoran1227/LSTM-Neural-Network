from Layers.Base import base_layer
import numpy as np
import math

class TanH(base_layer):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        # f(x)= (e^x-e^-x)/(e^x+e^-x)
        self.activation = np.divide(np.power(math.e , input_tensor) - np.power(math.e , -1 * input_tensor),
                                    np.power(math.e , input_tensor) + np.power(math.e , -1 * input_tensor))
        return self.activation

    def backward(self, error_tensor):
        # f'(x) = 1 - (f(x))^2
        error_tensor = error_tensor * (1 - np.power(self.activation, 2))
        return error_tensor