import numpy as np
from Layers.Base import base_layer

class SoftMax(base_layer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        #in order to avoid "Not a number" fault, we need do normalization
        norm = np.amax(input_tensor,axis=1)
        input_tensor = input_tensor - norm.reshape(-1,1)

        #softmax realization
        batch_sum = np.sum(np.exp(input_tensor), axis = 1)          #the sum of exponential units value in every element of batch
        self.output_tensor = np.divide(np.exp(input_tensor), batch_sum.reshape(-1,1))
        return np.copy(self.output_tensor)

    def backward(self, error_tensor):
        # e=a*(e-sum)
        error_sum = np.sum(error_tensor*self.output_tensor,axis = 1)
        error_tensor = self.output_tensor * (error_tensor - error_sum.reshape(-1,1))
        return error_tensor
