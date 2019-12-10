import numpy as np
import math
import scipy.signal as sgl
import copy

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.H = num_kernels            #number of kernels
        self.C = convolution_shape[0]   #number of channels
        self.M = convolution_shape[1]   #kenel shape
        self.B = None                   #batch_size
        self.Y = None                   #input_image rows
        self.X = None                   #input_image columns
        #determine the layer is 1D or 2D
        if(len(convolution_shape) == 2):    #if the layer is 1D
            self.flag = "1D"
            self.N = 1                  #kernel shape
            self.stride = (stride_shape[0], 1)
            # when stride is a single value, it's type is list and I make it a tuple with X_stride=1!!can not be 0!!
        elif(len(convolution_shape) == 3):  #if the layer is 2D
            self.flag = "2D"
            self.N = convolution_shape[2]
            if len(stride_shape) == 1:
                self.stride = (stride_shape[0],stride_shape[0])
            else:
                self.stride = stride_shape              #stride is a tuple
        else:
            print("wrong input size")

        #initialization of weights and bias
        self.weights = np.random.rand(self.H, self.C, self.M, self.N)   # W (H,c,m,n)
        self.bias = np.random.rand(self.H, 1)                           # B (H,1)

        self.grad_weights = None        #store the gradient of weights
        self.grad_bias = None           #store the gradient of bias
        self._optimizer = None
        self.weights_optimizer = None
        self.bias_optimizer = None
        self.input_tensor = None        #store input_tensor for backward propagation   shape [B,C,Y,X]

# After initialization, W (H,C,M,N), B (H,1), stride_shape(Y_stride, X_stride)

    def forward(self, input_tensor):
        self.B = input_tensor.shape[0]
        self.Y = input_tensor.shape[2]
        if self.flag =="1D":        #if 1D case
            # if input is 1D, make it has shape (B,C,Y,1)
            self.X = 1
            self.input_tensor = input_tensor.reshape(self.B, self.C, self.Y, self.X)
            output_shape =(self.B, self.H, math.ceil(self.Y / self.stride[0]))     #output_shape:(B,H,Y)!!!!! corresponds to input_tensor
        else:       #if 2D case
            # we don't need to reshape(normalize) input_tensor, then calculate the shape of output (B,H,Y',X')
            self.X = input_tensor.shape[3]
            self.input_tensor = input_tensor
            output_shape = (self.B, self.H, math.ceil(self.Y / self.stride[0]), math.ceil(self.X / self.stride[1]))    #output_shape:(B,H,Y',X')

        # After generalization of 1D case,
        # input_tensor shape is (B,C,Y,X), weights shape is (H,C,M,N), stride is (Y_stride, X_stride)

#########################################################################################################
        # # Forward propagation by subsampling after convolution
        #
        # num_Y = math.ceil(self.Y / self.stride[0])  #number of convolution centers in Y axis
        # num_X = math.ceil(self.X / self.stride[1])  #number of convolution centers in X axis
        # output_tensor = np.zeros((self.B, self.H, num_Y, num_X))
        # for b in range(self.B):  # in each batch_element
        #     for h in range(self.H):  # in each convolutional kernel
        #         tmp = 0     # stores the result of convolution without subsampling
        #         for c in range(self.C):  # in each channel
        #             tmp += sgl.correlate2d(self.input_tensor[b, c, :, :], self.weights[h, c, :, :], "same")
        #         #subsampling, the positions which order can divide stride exactly(整除) should remain.
        #         output_tensor[b, h, :, :] = tmp[: : self.stride[0], : : self.stride[1]]
        #         #add bias
        #         output_tensor[b, h, :, :] += self.bias[h]
        # #for now, output_tensor shape is (B, H, num_Y, num_X). reshape it for 1D case
        # output_tensor = output_tensor.reshape(output_shape)
##########################################################################################################
##########################################################################################################
        # Forward propagation by im2col

        num_Y = math.ceil(self.Y / self.stride[0])  #number of convolution centers in Y axis
        num_X = math.ceil(self.X / self.stride[1])  #number of convolution centers in X axis
        b_col = self.M // 2                         # before_pad_width in M orientation
        a_col = (self.M - 1) // 2                   # after_pad_width in M orientation
        b_row = self.N // 2                         # before_pad_width in N orientation
        a_row = (self.N - 1) // 2                   # after_pad_width in N orientation
        output_tensor = np.zeros((self.B, self.H, num_Y, num_X))
        for b in range(self.B):  # in each batch_element
            for h in range(self.H):  # in each convolutional kernel
                for c in range(self.C):  # in each channel
                    img = self.input_tensor[b, c, :, :]
                    img = np.pad(img, ((b_col, a_col), (b_row, a_row)), "constant", constant_values=0)  #padding
                    im2col = []
                    for i in range(num_Y):
                        o_Y = i * self.stride[0]        # Y_coordinate of origin of convolutional region
                        for j in range(num_X):
                            o_X = j * self.stride[1]    # X_coordinate of origin of convolutional region
                            im2col.append(img[o_Y: o_Y + self.M, o_X: o_X + self.N])
                    img = np.array(im2col).reshape(num_Y * num_X, self.M * self.N)      # img has been converted to convolution region columns
                    convolution = np.dot(img, self.weights[h, c, :, :].reshape(-1, 1))  # calculate convolution
                    output_tensor[b, h, :, :] += convolution.reshape(num_Y, num_X)      # superposition of different channels
                # add bias
                output_tensor[b, h, :, :] += self.bias[h]
        output_tensor = output_tensor.reshape(output_shape)
##########################################################################################################
        return output_tensor

    def backward(self, error_tensor):
        #Generalization of 1D case and calculate output error shape
        if self.flag == "1D":
            error_tensor = error_tensor.reshape(self.B, self.H, error_tensor.shape[2], 1)  #(B,H,Y',1)
            out_error_shape = (self.B, self.C, self.Y)  #(B,C,Y)
        else:
            out_error_shape = self.input_tensor.shape   #(B,C,Y,X)

        #error_tensor (B,H,Y',X') output_error_shape (B,C,Y,X) input_tensor (B,C,Y,X) weights (H,C,M,N)

        #backward propagation by upsampling before convolution

        # preparation
        b_col = self.M // 2          # before_pad_width in M orientation
        a_col = (self.M - 1) // 2    # after_pad_width in M orientation
        b_row = self.N // 2          # before_pad_width in N orientation
        a_row = (self.N - 1) // 2    # after_pad_width in N orientation
        output_error = np.zeros_like(self.input_tensor)  # stores the output_error (B,C,Y,X)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        for b in range(self.B):  # in each batch_element
            for h in range(self.H):  # in each convolutional kernel
                # upsampling
                E_h = np.zeros((self.Y, self.X))  # E_h records the upsampling error_tensor
                E_h[ : : self.stride[0], : : self.stride[1]] = error_tensor[b, h, :, :]

                for c in range(self.C):  # in each channel
                    #padding for a_c in the same way as forward propagation in order to calculate gradients
                    a_c = self.input_tensor[b, c, :, :]     #the input image of c_th channel
                    a_c_padding = np.pad(a_c, ((b_col, a_col), (b_row, a_row)), "constant", constant_values=0)

                    #Calculation of error_tensor which is backward propagated to next layer
                    #sgl.convolve2d include the rotation of filter, sgl.correlate2d does not rotate filter
                    output_error[b, c, :, :] += sgl.convolve2d(E_h, self.weights[h, c, :, :], "same")

                    #calculate gradients of weights and bias
                    self.grad_weights[h, c, :, :] += sgl.correlate2d(a_c_padding, E_h, "valid")

                self.grad_bias[h] += np.sum(E_h)
        output_error = output_error.reshape(out_error_shape)

        #gradients update
        if self.weights_optimizer is not None:
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.grad_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.grad_bias)

        return output_error

    def initialize(self, weights_initializer, bias_initializer):    #fucntion which can reinitialize weights and bias
        #fan_in = c*m*n   fan_out=H*m*n
        fan_in = self.C * self.M * self.N
        fan_out = self.H * self.M * self.N
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def gradient_weights(self):
        return self.grad_weights

    @property
    def gradient_bias(self):
        return self.grad_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,opt):
        self._optimizer = opt
        self.weights_optimizer =copy.deepcopy(self._optimizer)   #setting weights_optimizer can not be achieved in __init__()
        self.bias_optimizer = copy.deepcopy(self._optimizer)
