import numpy as np

class Constant:
    def __init__(self, constant = 0.1):                    #default value is 0.1
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):           #为了保证后续调用initializer时的一致性，这两个未用到的函数参数不能删去
        init_tensor = np.zeros(weights_shape) + self.constant       #all the initial weights are set with constant value
        return init_tensor

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        init_tensor = np.random.uniform(0, 1, weights_shape)    #initialize weights by uniform distribution
        return init_tensor

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):           #fan_in and fan_out is two values which indicates the number of input oroutput nodes
        variance = np.sqrt(2 / (fan_in + fan_out))
        init_tensor = np.random.normal(0.0, variance, weights_shape)
        return init_tensor

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2 / fan_in)
        init_tensor = np.random.normal(0.0, variance, weights_shape)
        return init_tensor
