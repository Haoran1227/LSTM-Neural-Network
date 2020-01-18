from Layers.Base import base_layer
from Layers.Helpers import compute_bn_gradients
import numpy as np
import copy

class BatchNormalization(base_layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels    # the number of channels of input_tensor in both the vector and image case
        self.weights = np.ones((1, self.channels))   # gamma, trainable shift parameter of BN. shape:(1,H)
        self.bias = np.zeros((1, self.channels))     # beta, trainable scale parameter of BN. shape:(1,H)
        self._optimizer = None
        self.weights_optimizer = None
        self.bias_optimizer = None
        self.grad_weights = None
        self.grad_biases = None

        # stores the shape of convolutional layer output for inverse transform in reformat function
        self.input_shape = None

        # member variables determined in forward pass and reused in backward pass
        self.flag = None            # can be "vector" or "image", it indicates the BN layer is after convolutional layer or fully-connected layer
        self.norm_tensor = None
        self.mean = None
        self.var = None
        self.input_tensor = None    #after reformat

        # member variables determined in training phase and used in test phase
        self.mean_estimate = None
        self.var_estimate = None
        self.alpha = 0.8                # moving average decay

    def initialize(self, weights_initializer, bias_initializer):
        # Ignores assigned initializers and initialize weights as ones and biases as zeros
        # because we don't want weights and bias of BN have an impact at the beginning of training
        self.weights = np.ones((1, self.channels))      #(1,H)
        self.bias = np.zeros((1, self.channels))        #(1,H)

    def reformat(self, tensor):
        # this function can do the transform and inverse transform between image-like tensor and vector-like tensor
        # in order to make BN layer adapt to convolutional case
        B, H, M, N = self.input_shape
        if len(tensor.shape) == 4:          # convert image-like tensor to vector-like tensor
            # image-like tensor (B,H,M,N)
            tensor = tensor.reshape(B, H, M*N)      #(B,H,MN)
            variant = np.zeros((B, M*N, H))         #(B,MN,H) stores the tensor after transpose
            for b in range(B):
                variant[b, :, :] = np.transpose(tensor[b, :, :])
            variant = variant.reshape(B*M*N, H)     #(BMN,H)
        else:                               #convert vector-like tensor to image-like tensor
            # vector-like tensor (B*M*N, H)
            tensor = tensor.reshape((B, M*N, H))      #(B,MN,H)
            variant = np.zeros((B, H, M*N))         #(B,H,MN)
            for b in range(B):
                variant[b, :, :] = np.transpose(tensor[b, :, :])
            variant = variant.reshape((B, H, M, N)) #(B,H,M,N)
        return variant

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        if len(input_tensor.shape) == 4:    # if BN after convolutional layer, need to convert 4D-input to 2D
            self.flag = 'image'
            self.input_tensor = self.reformat(input_tensor)
        else:
            self.flag = 'vector'
            self.input_tensor = input_tensor

        if self.phase == "train":
            # calculate mean and variance of current batch
            # input_tensor is (BMN,H) or (B,H), we need to calculate mean and variance through BMN or B axis
            self.mean = np.mean(self.input_tensor, axis=0).reshape(1,-1)     # mean vector (1,H)
            self.var = np.var(self.input_tensor, axis=0).reshape(1,-1)       # variance vector (1,H)

            # moving average estimation of training set mean and variance
            if self.mean_estimate is None:      #initialization of mean_estimate
                self.mean_estimate = self.mean
            if self.var_estimate is None:       #initialization of var_estimate
                self.var_estimate = self.var
            # online estimation during training phase by moving average
            self.mean_estimate = self.alpha * self.mean_estimate + (1-self.alpha) * self.mean
            self.var_estimate = self.alpha * self.var_estimate + (1-self.alpha) * self.var

        if self.phase == "test":
            # in test phase, use the mean and variance of all the training set as that of test set
            self.mean = self.mean_estimate
            self.var = self.var_estimate

        # Normalization
        self.norm_tensor = (self.input_tensor - self.mean) / np.sqrt(self.var + np.finfo(float).eps)

        # Scale and shift
        output_tensor = self.weights * self.norm_tensor + self.bias

        # modify the output in image-tensor case
        if self.flag == "image":        #if input is 4D, we need convert output to 4D
            output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self, error_tensor):
        if self.flag =="image":
            error_tensor = self.reformat(error_tensor)

        # error_tensor shape (B,H) or (BMN,H)
        # calculation of gradients
        self.grad_weights = np.sum(error_tensor * self.norm_tensor, axis = 0).reshape(1,-1)
        self.grad_biases = np.sum(error_tensor, axis = 0).reshape(1,-1)

        # update weights and bias
        if self.weights_optimizer is not None:
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.grad_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.grad_biases)


        #calculation of output_error
        output_error = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var)
        if self.flag =="image":
            output_error = self.reformat(output_error)

        return output_error

    @property
    def gradient_weights(self):
        return self.grad_weights

    @property
    def gradient_bias(self):
        return self.grad_biases

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,opt):
        self._optimizer = opt
        self.weights_optimizer =copy.deepcopy(self._optimizer)   #setting weights_optimizer can not be achieved in __init__()
        self.bias_optimizer = copy.deepcopy(self._optimizer)
        self.bias_optimizer.regularizer = None      # Bias shouldn't participate in regularization

    @property
    def regularization_loss(self):
        loss = 0
        if self.weights_optimizer is not None:  # if weights_optimizer is defined
            if self.weights_optimizer.regularizer is not None:  #if weights_optimizer has regularizer
                # calculate regularization_loss
                loss = self.weights_optimizer.regularizer.norm(self.weights)
        return loss
