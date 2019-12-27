import numpy as np
import copy
from Layers.Base import base_layer
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

class LSTM_cell:
    """
    This class defines one LSTM-cell which is used to build a LSTM layer.
    It includes the realization of forward and backward propagation algorithm
    Attributes:
        W_xh, W_hh, B_h, W_hy, B_y: trainable parameters of LSTM-cell. Because LSTM layer shares
            weights in sequential LSTM cells, we need to make all LSTM cells in the same layer have
            the same weights and the update of weights needs to be implemented after calculating all
            gradients of cells.
        sigmoid: store sigmoid activation functions so that the variables can be transferred from
            forward pass to backward pass
        tanh: store tanh activation functions
        cache: store the variables in forward pass for the gradient calculation in backward pass.
    """
    def __init__(self, W_xh, W_hh, B_h, W_hy, B_y):
        # W_xh = [W_xf W_xi W_xc W_xo].T    shape:(4H, J)   each W:(H, J)
        # W_hh = [W_hf W_hi W_hc W_ho].T    shape:(4H, H)
        # B_h = [B_xf B_xi B_xc B_xo]       shape:(1, 4H)   each B:(1, H)
        # W_hy  shape:(K, H)
        # B_y   shape:(1, K)
        self.W_xh, self.W_hh, self.B_h, self.W_hy, self.B_y = W_xh, W_hh, B_h, W_hy, B_y

        # Variables which are stored in forward pass for backward pass
        self.sigmoid = []            # store sigmoid activation functions
        self.tanh = []               # store tanh activation functions
        self.cache = None            # store variables used in calculating gradients in backward pass

    def forward(self, input_tensor, hidden_state, cell_state):
        """
        This function realize the forward propagation of one LSTM-cell
        Args:
            input_tensor: x_t, input tensor of current slot.
            hidden_state: h_t-1, hidden state of previous LSTM-cell (time slot)
            cell_state: c_t-1, cell state of previous LSTM-cell (time slot)
        Returns:
            ndarray, output_tensor: y_t, output tensor of current slot.
            ndarray, next_h: h_t, hidden state of previous LSTM-cell, which will be transferred to next cell
            ndarray, next_c: c_t, cell state of previous LSTM-cell, which will be transferred to next cell
        """
        # Data preparation.
        x = input_tensor.reshape(1, -1)   # shape: (1,J)
        prev_h = hidden_state             # shape: (1,H)
        prev_c = cell_state               # shape: (1,H)
        _, H = hidden_state.shape         # hidden size

        # initialize tanh and sigmoid functions
        for _ in range(4):
            sigmoid = Sigmoid()
            self.sigmoid.append(sigmoid)
        for _ in range(2):
            tanh = TanH()
            self.tanh.append(tanh)

        # forward propagation in LSTM-cell
        embedding = np.dot(x, self.W_xh.T) + np.dot(prev_h, self.W_hh.T) + self.B_h     # (1,4H)
        f = self.sigmoid[0].forward(embedding[:, :H])
        i = self.sigmoid[1].forward(embedding[:, H:2*H])
        c_hat = self.tanh[0].forward(embedding[:, 2*H:3*H])
        o = self.sigmoid[2].forward(embedding[:, 3*H:])
        # calculation of new cell_state
        next_c = prev_c * f + i * c_hat
        # calculation of new hidden_state
        tanh_output = self.tanh[1].forward(next_c)
        next_h = o * tanh_output
        # calculation of output
        output_tensor = self.sigmoid[3].forward(np.dot(next_h, self.W_hy.T) + self.B_y)

        # return the variables which are needed in backward propagation
        self.cache = [f, i, c_hat, o, x, prev_h, prev_c, tanh_output, next_h, next_c]

        return output_tensor, next_h, next_c

    def backward(self, error_tensor, dnext_c, dnext_h):
        """
        This function realize the backward propagation of one LSTM-cell
        Args:
            error_tensor: e_t, input error tensor of current LSTM-cell
            dnext_c: input cell_state error, input error tensor refer to cell state from next LSTM-cell
            dnext_h: input hidden_state error, input error tensor refer to hidden state from next LSTM-cell
        Returns:
            dx, dprev_c, dprev_h, dW_xh, dW_hh, dB_h, dW_hy, dB_y
            # the derivatives of output to previous LSTM-cell and previous layer and trainable parameters
        """
        # load stored variables in forward pass.
        f, i, c_hat, o, x, prev_h, prev_c, tanh_output, next_h, next_c = self.cache

        # calculation of gradients of W_hy and B_y
        error_tensor = self.sigmoid[3].backward(error_tensor)
        dW_hy = np.dot(error_tensor.reshape(-1, 1), next_h.reshape(1, -1))  # shape: (K, H)
        dB_y = error_tensor  # shape: (1, K)

        # backward propagation of error
        error_tensor = np.dot(error_tensor, self.W_hy)
        dnext_h += error_tensor
        do = self.sigmoid[2].backward(dnext_h * tanh_output)
        dnext_c += self.tanh[1].backward(dnext_h * o)
        dprev_c = dnext_c * f
        df = self.sigmoid[0].backward(dnext_c * prev_c)
        dc_hat = self.tanh[0].backward(dnext_c * i)
        di = self.sigmoid[1].backward(dnext_c * c_hat)

        # calculation of gradients
        dmix = np.concatenate((df, di, dc_hat, do), axis=1)     #(1, 4H)
        dx = np.dot(dmix, self.W_xh)
        dprev_h = np.dot(dmix, self.W_hh)
        dW_xh = np.dot(dmix.reshape(-1, 1), x.reshape(1, -1))
        dW_hh = np.dot(dmix.reshape(-1, 1), prev_h.reshape(1, -1))
        dB_h = dmix

        return dx, dprev_c, dprev_h, dW_xh, dW_hh, dB_h, dW_hy, dB_y



class LSTM(base_layer):
    """
    This class build LSTM layer based on LSTM-cell class.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.J, self.H, self.K = input_size, hidden_size, output_size

        # initialize trainable parameters
        self.W_xh = np.random.rand(4*self.H, self.J)    # W_xh = [W_xf W_xi W_xc W_xo].T    shape:(4H, J)   W-(H, J)
        self.W_hh = np.random.rand(4*self.H, self.H)    # W_hh = [W_hf W_hi W_hc W_ho].T    shape:(4H, H)
        self.B_h = np.random.rand(1, 4*self.H)          # B_h = [B_xf B_xi B_xc B_xo]       shape:(1, 4H)   B-(1, H)
        self.W_hy = np.random.rand(self.K, self.H)      # W_hy  shape:(K, H)
        self.B_y = np.random.rand(1, self.K)            # B_y   shape:(1, K)

        self._memory = False    # indicates whether subsequent batch sequence has relation with the last one

        # initialize the hidden_state and cell_state for time slot 0.
        self.hidden_state = np.zeros((1, self.H))
        self.cell_state = np.zeros((1, self.H))

        # initialize gradients
        self.grad_weights = None

        # initialize optimizers
        self._optimizer = None
        self.W_xh_optimizer, self.W_hh_optimizer, self.W_hy_optimizer, \
        self.B_y_optimizer, self.B_h_optimizer= None, None, None, None, None

        # bulid LSTM layer using LSTM-cells
        self.layer = []

    def forward(self, input_tensor):
        """
        This function implement forward propagation of LSTM layer with help of the forward
        propagation function of LSTM-cell
        Args:
            input_tensor: input tensor of LSTM layers, its shape should be (batch_size, input_size)
        Returns:
            output_tensor: output tensor of LSTM layers, its shape is (batch_size, output_size)
        """
        batch_size = input_tensor.shape[0]      # batch size: input_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.K))

        # determine whether sequential batches have relation. If memory is true, the hidden_state
        # and cell_state of last cell will be transferred to the next cell
        if not self._memory:  # if the next batch(time batch) has no relation with the previous batch
            self.hidden_state = np.zeros((1, self.H))   # initialize hidden_state as 0
            self.cell_state = np.zeros((1, self.H))     # initialize cell_state as 0

        # build LSTM layer based on LSTM cells
        for t in range(batch_size):
            cell = LSTM_cell(self.W_xh, self.W_hh, self.B_h, self.W_hy, self.B_y)
            output_tensor[t, :], self.hidden_state, self.cell_state = \
                cell.forward(input_tensor[t, :], self.hidden_state, self.cell_state)
            self.layer.append(cell)

        return output_tensor

    def backward(self, error_tensor):
        """
        This function implement backward propagation of LSTM layer with help of the backward
        propagation function of LSTM-cell
        Args:
            error_tensor: error of LSTM layers which is transferred from next layer, shape should be (B,K)
        Returns:
            output_error: output error transferred from LSTM layers to last layer, shape is (batch_size, output_size)
        """

        # initialization
        # initialize output_error
        batch_size = error_tensor.shape[0]
        output_error = np.zeros((batch_size, self.J))
        # initialize hidden_error tensor and cellstate_error for last layer
        dnext_c, dnext_h = np.zeros((1, self.H)), np.zeros((1, self.H))
        # initialization of gradients
        # W_xh:(4H, J)  W_hh:(4H, H)  B_h:(1, 4H)  W_hy:(K, H)  B_y:(1, K)
        grad_W_xh, grad_W_hh, grad_W_hy, grad_B_h, grad_B_y = \
            np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy), np.zeros_like(self.B_h), np.zeros_like(self.B_y)

        # calculation of hidden_error and output error
        for t in reversed(range(batch_size)):
            output_error[t, :], dnext_c, dnext_h, dW_xh, dW_hh, dB_h, dW_hy, dB_y =\
                self.layer[t].backward(error_tensor[t, :], dnext_c, dnext_h)

            # accumulation of gradients of LSTM-cells
            grad_W_xh += dW_xh
            grad_W_hh += dW_hh
            grad_B_h += dB_h
            grad_W_hy += dW_hy
            grad_B_y += dB_y

        # stores gradients of hidden units for Unitest
        self.grad_weights = np.concatenate((grad_W_xh, grad_W_hh, grad_B_h.reshape(-1, 1)), axis=1)

        # update weights and biases
        if self._optimizer is not None:
            self.W_xh = self.W_xh_optimizer.calculate_update(self.W_xh, grad_W_xh)
            self.W_hh = self.W_hh_optimizer.calculate_update(self.W_hh, grad_W_hh)
            self.W_hy = self.W_hy_optimizer.calculate_update(self.W_hy, grad_W_hy)
            self.B_h = self.B_h_optimizer.calculate_update(self.B_h, grad_B_h)
            self.B_y = self.B_y_optimizer.calculate_update(self.B_y, grad_B_y)

        # After one propagation process, the memory flag will be re-initialized.
        self._memory = False
        return output_error

    # initialization of weights and biases
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

    # property of weights, which is used to make LSTM compatible with other layers
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
        # W_xh: (4H, J)   W_hh: (4H, H)    B_h: (1, 4H)
        if weights_hidden is not None:
            self.W_xh = weights_hidden[:, :self.J]
            self.W_hh = weights_hidden[:, self.J:-1]
            self.B_h = weights_hidden[:, -1].reshape(1, 4*self.H)
        else:
            self.W_xh = None
            self.W_hh = None
            self.B_h = None

    # property to get gradients
    @property
    def gradient_weights(self):
        return self.grad_weights

    # property to get and set optimizers
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
