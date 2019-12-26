import numpy as np
import copy
from Layers.Base import base_layer
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

class RNN_cell:
    def __init__(self, W_xh, W_hh, W_hy, B_h, B_y):
        # W_xh:(H, J)   W_hh: (H, H)    W_hy: (K, H)    B_h: (1, H)    B_y: (1, K)
        self.W_xh, self.W_hh, self.W_hy, self.B_h , self.B_y = W_xh, W_hh, W_hy, B_h, B_y

        # Variables which are stored in forward pass for backward pass
        self.sigmoid = None          # store tanh activation function
        self.tanh = None             # store tanh activation function
        self.input_tensor = None     # store input_tensor
        self.output_h = None        # hidden state of current cell
        self.input_h = None         # hidden state of last cell

    def forward(self, input_tensor, hidden_state):
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        # store variables which are needed in backward pass
        self.input_tensor = input_tensor
        self.input_h = hidden_state
        # forward propagation algorithm
        self.output_h = self.tanh.forward(np.dot(hidden_state, self.W_hh.T) +
                                    np.dot(input_tensor, self.W_xh.T) + self.B_h)
        output_tensor = self.sigmoid.forward(np.dot(self.output_h, self.W_hy.T) + self.B_y)
        return output_tensor, self.output_h

    def backward(self, error_tensor, hidden_error):
        error_tensor = self.sigmoid.backward(error_tensor)
        e_tmp = self.tanh.backward(np.dot(error_tensor, self.W_hy) + hidden_error)  # error transferred over tanh
        hidden_error = np.dot(e_tmp, self.W_hh)
        output_error = np.dot(e_tmp, self.W_xh)
        grad_W_hy = np.dot(error_tensor.reshape(-1, 1), self.output_h.reshape(1, -1))  # grad_V (K, H)
        grad_B_y = error_tensor  # (1, K)
        grad_W_hh = np.dot(e_tmp.reshape(-1, 1), self.input_h.reshape(1, -1))  # (H, H)
        grad_W_xh = np.dot(e_tmp.reshape(-1, 1), self.input_tensor.reshape(1, -1))  # (H, J)
        grad_B_h = e_tmp
        return output_error, hidden_error, grad_W_hy, grad_B_y, grad_W_hh, grad_W_xh, grad_B_h

class RNN(base_layer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.H, self.J, self.K = hidden_size, input_size, output_size

        # W_xh:(H, J)   W_hh: (H, H)    W_hy: (K, H)
        self.W_xh = np.random.rand(self.H, self.J)
        self.W_hh = np.random.rand(self.H, self.H)
        self.W_hy = np.random.rand(self.K, self.H)
        # B_h: (1, H)    B_y: (1, K)
        self.B_h = np.random.rand(1, self.H)
        self.B_y = np.random.rand(1, self.K)

        self._memory = False    # indicates whether subsequent batch sequence has relation with the last one

        # the hidden_state for time slot 0.
        # it needs to be a global variable, because it transfers hidden_state between batch.
        self.hidden_state = np.zeros((1, self.H))

        # initialize gradients
        self.grad_weights = None

        # initialize optimizers
        self._optimizer = None
        self.W_xh_optimizer, self.W_hh_optimizer, self.W_hy_optimizer,\
        self.B_h_optimizer, self.B_y_optimizer = None, None, None, None, None

        # bulid RNN layer using RNN-cells
        self.layer = []

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]  # batch size: input_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.K))

        if not self._memory:  # if the next batch(time batch) has no relation with the last batch
            self.hidden_state = np.zeros((1, self.H))  # initialize hidden_state as 0

        for t in range(batch_size):
            cell = RNN_cell(self.W_xh, self.W_hh, self.W_hy, self.B_h, self.B_y)
            output_tensor[t, :], self.hidden_state = cell.forward(input_tensor[t, :], self.hidden_state)
            self.layer.append(cell)

        return output_tensor

    def backward(self, error_tensor):
        # error_tensor (B,K)  output_error (B,J)
        # initialize output_error
        batch_size = error_tensor.shape[0]
        output_error = np.zeros((batch_size, self.J))
        # initialize hidden_error tensor for last layer
        hidden_error = np.zeros((1, self.H))
        # initialization of gradients
        # W_xh:(H, J)   W_hh: (H, H)    W_hy: (K, H)    B_h: (1, H)    B_y: (1, K)
        grad_W_xh = np.zeros_like(self.W_xh)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_W_hy = np.zeros_like(self.W_hy)
        grad_B_h = np.zeros_like(self.B_h)
        grad_B_y = np.zeros_like(self.B_y)

        # calculation of hidden_error and output error
        for t in reversed(range(batch_size)):
            output_error[t, :], hidden_error, d_W_hy, d_B_y, d_W_hh, d_W_xh, d_B_h =\
                self.layer[t].backward(error_tensor[t, :], hidden_error)

            grad_W_hy += d_W_hy
            grad_B_y += d_B_y
            grad_W_hh += d_W_hh
            grad_W_xh += d_W_xh
            grad_B_h += d_B_h

        # stores gradients of hidden unit for Unitest
        self.grad_weights = np.concatenate((grad_W_xh, grad_W_hh, grad_B_h.reshape(-1,1)), axis=1)

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
        # for W_xh, B_h: fan_in = input_size   fan_out=hidden_size
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
        # weights must be [W_xh, W_hh, B_h.T] in order to correspond to Unitest.
        # In Helpers.py, use "print(it.multi_index)  print(analytical_derivative - numerical_derivative)" to check order of weights
        weights_hidden = np.concatenate((self.W_xh, self.W_hh, self.B_h.T), axis=1)
        return weights_hidden

    @weights.setter
    def weights(self, weights_hidden):
        # Because the base layer has weights attribute and initialized as None.
        # We need to determine whether weights is None, otherwise the initialization of RNN has fault.
        # W_xh:(H, J)   W_hh: (H, H)    B_h: (1, H)
        if weights_hidden is not None:
            self.W_xh = weights_hidden[:, :self.J]
            self.W_hh = weights_hidden[:, self.J:-1]
            self.B_h = weights_hidden[:, -1].reshape(1, self.H)
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
