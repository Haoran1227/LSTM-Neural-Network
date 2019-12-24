from Layers.Base import base_layer
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import numpy as np

class LSTM(base_layer):
    def __init__(self, input_size, hidden_size, output_size, bptt_length):
        super().__init__()
        self.H = hidden_size
        self.J = input_size
        self.K = output_size
        self.bptt_length = bptt_length

        # input_tensor: (1, J)  output_tensor: (1,K)    hidden_tensor: (1,H)
        self.hidden_state = np.zeros((1, self.H))  # initialize as 0 for time 0

        self._memory = False


    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass

    def initialize(self, weights_initializer, bias_initializer):
        pass

    @property
    def memorize(self):
        return self._memory

    @memorize.setter
    def memorize(self, mem):
        self._memory = mem