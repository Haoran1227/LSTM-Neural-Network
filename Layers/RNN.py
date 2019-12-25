import numpy as np
import copy
from Layers.Base import base_layer
from Layers.TanH import TanH

class RNN(base_layer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.H = hidden_size
        self.J = input_size
        self.K = output_size

        # input_tensor: (1, J)  output_tensor: (1,K)    hidden_tensor: (1,H)
        self.hidden_state = np.zeros((1, self.H))       #initialize as 0 for time 0

        # W_xh:(H, J)   W_hh: (H, H)    W_hy: (K, H)
        self.W_xh = np.random.rand(self.H, self.J)
        self.W_hh = np.random.rand(self.H, self.H)
        self.W_hy = np.random.rand(self.K, self.H)

        # B_h: (1, H)    B_y: (1, K)
        self.B_h = np.random.rand(1, self.H)
        self.B_y = np.random.rand(1, self.K)

        # Variables which are stored in forward pass for backward pass
        self.tanh = []          #store tanh function in each hidden cell
        self.input_tensor = None

        self._memory = False    # indicates whether subsequent batch sequence has relation with the last one

        #initialize gradients
        self.grad_weights = None

        # initialize optimizers
        self._optimizer = None
        self.W_xh_optimizer = None
        self.W_hh_optimizer = None
        self.W_hy_optimizer = None
        self.B_h_optimizer = None
        self.B_y_optimizer = None

    def forward(self, input_tensor):
        # input_tensor shape (B, J)
        if not self._memory:   # if the next batch(time batch) has no relation with the last batch
            self.tanh = []                                  #initialize tanh list
            self.hidden_state = np.zeros((1, self.H))       #initialize hidden state as zero again
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]                  # batch size: input_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.K))
        for t in range(batch_size):
            # creates tanh function for the current cell
            tanh = TanH()
            # forward propagation algorithm
            hidden_state = tanh.forward(np.dot(self.hidden_state[-1, :], self.W_hh.T) +
                                        np.dot(input_tensor[t, :], self.W_xh.T) + self.B_h)
            output_tensor[t, :] = np.dot(hidden_state, self.W_hy.T) + self.B_y
            # stores the information of activation of tanh for backward propagation
            self.hidden_state = np.vstack((self.hidden_state, hidden_state))    #stack the new state at end of hidden_state array
            self.tanh.append(tanh)
        return output_tensor

    def backward(self, error_tensor):
        # error_tensor (B,K)  hidden_error (B,H)
        # initialize hidden_error and output_error
        batch_size = error_tensor.shape[0]
        hidden_error = np.zeros((batch_size, self.H))
        output_error = np.zeros((batch_size, self.J))

        # calculation of hidden_error and output error
        hidden_error[-1, :] = self.tanh[-1].backward(np.dot(error_tensor[-1, :], self.W_hy))
        output_error[-1, :] = np.dot(hidden_error[-1, :], self.W_xh)
        for t in reversed(range(batch_size-1)):
            hidden_error[t, :] = self.tanh[t].backward(np.dot(error_tensor[t, :], self.W_hy) +
                                                       np.dot(hidden_error[t+1, :], self.W_hh))
            output_error[t, :] = np.dot(hidden_error[t, :], self.W_xh)

        # initialization of gradients
        # W_xh:(H, J)   W_hh: (H, H)    W_hy: (K, H)    B_h: (1, H)    B_y: (1, K)
        grad_W_xh = np.zeros_like(self.W_xh)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_W_hy = np.zeros_like(self.W_hy)
        grad_B_h = np.zeros_like(self.B_h)
        grad_B_y = np.zeros_like(self.B_y)

        # calculation of gradients
        # self.hidden_state (B+1, H), because of its 0 time slot
        # np.dot() for (13,) and (7,)     ????
        for t in range(batch_size):
            grad_W_hy += np.dot(error_tensor[t, :].reshape(-1,1), self.hidden_state[t+1, :].reshape(1, -1))    #grad_V (K, H)
            grad_B_y += error_tensor[t, :]                                                                     #(1, K)
            grad_W_hh += np.dot(self.hidden_state[t, :].reshape(-1,1), hidden_error[t, :].reshape(1, -1))      #(H, H)
            grad_W_xh += np.dot(hidden_error[t, :].reshape(-1,1), self.input_tensor[t, :].reshape(1, -1))      #(H, J)
            grad_B_h += hidden_error[t, :]                                                               #(1, H)

        # stores gradients of hidden unit for Unitest
        self.grad_weights = np.concatenate((grad_W_xh, grad_W_hh.T, grad_B_h.reshape(-1,1)), axis=1)

        # update weights and biases
        if self._optimizer is not None:
            self.W_xh = self.W_xh_optimizer.calculate_update(self.W_xh, grad_W_xh)
            self.W_hh = self.W_hh_optimizer.calculate_update(self.W_hh, grad_W_hh)
            self.W_hy = self.W_hy_optimizer.calculate_update(self.W_hy, grad_W_hy)
            self.B_h = self.B_h_optimizer.calculate_update(self.B_h, grad_B_h)
            self.B_y = self.B_y_optimizer.calculate_update(self.B_y, grad_B_y)

        #在一次反向传播更新后，重置RNN的状态
        self._memory = False
        return output_error

    def initialize(self, weights_initializer, bias_initializer):    #fucntion which can reinitialize weights and bias
        # # for W_xh, B_h: fan_in = input_size   fan_out=hidden_size
        self.W_xh = weights_initializer.initialize(self.W_xh.shape, self.J, self.H)
        self.B_h = bias_initializer.initialize(self.B_h.shape, self.J, self.H)
        # for W_hh: fan_in = hidden_size   fan_out=hidden_size
        self.W_hh = weights_initializer.initialize(self.W_hh.shape, self.H, self.H)
        # for W_hy, B_y: fan_in = hidden_size   fan_out=output_size
        self.W_hy = weights_initializer.initialize(self.W_hy.shape, self.H, self.K)
        self.B_y = bias_initializer.initialize(self.B_y.shape, self.H, self.K)

    @property
    def memorize(self):
        return self._memory

    @memorize.setter
    def memorize(self, mem):
        self._memory = mem

    @property
    def weights(self):
        # weights only include W_xh, W_hh, B_h. It means the parameters needed for calculation of hidden state.
        # weights must be [W_xh, W_hh.T, B_h.T] in order to correspond to Unitest.
        # In Helpers.py, use "print(it.multi_index)  print(analytical_derivative - numerical_derivative)" to check order of weights
        weights_hidden = np.concatenate((self.W_xh, self.W_hh.T, self.B_h.T), axis=1)
        return weights_hidden

    @weights.setter
    def weights(self, weights_hidden):
        # Because the base layer has weights attribute and initialized as None.
        # We need to determine whether weights is None, otherwise the initialization of RNN has fault.
        # W_xh:(H, J)   W_hh: (H, H)    B_h: (1, H)
        if weights_hidden is not None:
            self.W_xh = weights_hidden[:, :self.J]
            self.W_hh = weights_hidden[:, self.J:-1]
            self.B_h = weights_hidden[:, -1]
        else:
            self.W_xh = None
            self.W_hh = None
            self.B_h = None

    @property
    def gradient_weights(self):
        return self.grad_weights

    # Some notes for property "optimizer" for RNN.
    # Refer to "NeuralNetwork.py" 21 line. Because the definition of weights in RNN is not
    # intact (W_hy, b not included),
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,opt):
        self._optimizer = opt
        self.W_xh_optimizer = copy.deepcopy(self._optimizer)
        self.W_hh_optimizer = copy.deepcopy(self._optimizer)
        self.W_hy_optimizer = copy.deepcopy(self._optimizer)
        self.B_h_optimizer = copy.deepcopy(self._optimizer)
        self.B_y_optimizer = copy.deepcopy(self._optimizer)
        # Bias should not participate in regularization
        self.B_h_optimizer.regularizer = None
        self.B_y_optimizer.regularizer = None

    @property
    def regularization_loss(self):
        if self._optimizer is not None:  # if weights_optimizer is defined
            if self._optimizer.regularizer is not None:  #if weights_optimizer has regularizer
                # calculate regularization_loss
                loss = self.W_xh_optimizer.regularizer.norm(self.W_xh) + \
                       self.W_hh_optimizer.regularizer.norm(self.W_hh) + \
                       self.W_hy_optimizer.regularizer.norm(self.W_hy)
            else:
                loss = 0
        else:
            loss = 0
        return loss
