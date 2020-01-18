import numpy as np
import copy
from Layers.Base import base_layer

class FullyConnected(base_layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(output_size, input_size+1)     #weights+biases: row number=output_size, column_number=input_size
        self.input_tensor = None                                     #A_(l-1)
        self.gradient = None                                         #gradient of weights and biases in one matrix
        self._optimizer = None
        self.weights_optimizer = None
        self.bias_optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        #the last column in weights is bias
        shape = self.weights.shape
        initial_weights= weights_initializer.initialize((shape[0],shape[1] - 1), shape[0], shape[1])
        initial_bias = bias_initializer.initialize((shape[0], 1), shape[0], shape[1])
        self.weights = np.concatenate((initial_weights, initial_bias), axis = 1)
        #self.weights[: , :-1] = weights_initializer.initialize((shape[0],shape[1] - 1), shape[0], shape[1])
        #self.bias[: , -1] = bias_initializer.initialize((shape[0], 1), shape[0], shape[1])
        #前一句可行，后一句不可行，因为只用-1得到的是最后一列的向量，形状是（3，）

    def forward(self,input_tensor):
        batch_size = input_tensor.shape[0]                                                      #input tensor is a matrix with columns of input size and rows of batch size
        self.input_tensor = np.concatenate((input_tensor, np.ones((batch_size,1))),axis = 1)    #add 1-vector in the end of input_tensor
        input_tensor = np.dot(self.input_tensor,self.weights.T)                                 #Z = np.dot(A, W.T)
        return input_tensor

    def backward(self,error_tensor):                                #error_tensor is the same shape as input_tensor
        # caculate gradients
        self.gradient = np.dot(error_tensor.T, self.input_tensor)

        #caculate the error tensor for next layer
        error_tensor = np.delete(np.dot(error_tensor, self.weights),-1,1)       # BP algorithm(Relu): E_l-1=E_l*W and delete the last column

        # update weights and biases
        if self._optimizer is not None:
            self.weights[:, :-1] = self.weights_optimizer.calculate_update(self.weights[:, :-1], self.gradient[:, :-1])
            self.weights[:, -1] = self.bias_optimizer.calculate_update(self.weights[:, -1], self.gradient[:, -1])
        return error_tensor

    #optimizer property attribute and gradient_weights property attribute
    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self,optimizer):
        self._optimizer = optimizer
        self.weights_optimizer = copy.deepcopy(self._optimizer)
        self.bias_optimizer = copy.deepcopy(self._optimizer)
        self.bias_optimizer.regularizer = None  # Bias should not participate in regularization

    optimizer = property(get_optimizer,set_optimizer)

    @property
    def gradient_weights(self):
        return self.gradient

    @property
    def regularization_loss(self):
        loss = 0
        if self._optimizer is not None:  # if weights_optimizer is defined
            if self._optimizer.regularizer is not None:  #if weights_optimizer has regularizer
                # calculate regularization_loss
                weights = np.delete(self.weights, -1, axis=1)   #delte the last column, i.e., the bias.
                loss = self._optimizer.regularizer.norm(weights)
        return loss