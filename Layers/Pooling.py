import numpy as np
from Layers.Base import base_layer

class Pooling(base_layer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        if len(pooling_shape) == 1:
            self.flag = "1D"
            self.M = pooling_shape[0]           #pooling filter shape: (M, N)
            self.N = 1
            self.stride = (stride_shape[0], 1)  #stride shape: (Y_stride, X_stride)
        else:
            self.flag = "2D"
            self.M = pooling_shape[0]
            self.N = pooling_shape[1]
            if len(stride_shape) == 1:
                self.stride = (stride_shape[0],stride_shape[0])
            else:
                self.stride = stride_shape

        self.input_shape = None             #stores input_tensor shape
        self.max_position = None            #stores position of maximum for backward propagation

    def forward(self, input_tensor):
        # Generalization of input_tensor in order to be compatible with 1D case
        if self.flag == "1D":
            B = input_tensor.shape[0]
            C = input_tensor.shape[1]
            Y = input_tensor.shape[2]
            X = 1
            input_tensor = input_tensor.reshape(Y, X)
            self.input_shape = input_tensor.shape          #input_tensor shape: (B,C,Y,X)
        else:
            B = input_tensor.shape[0]
            C = input_tensor.shape[1]
            Y = input_tensor.shape[2]
            X = input_tensor.shape[3]
            self.input_shape = input_tensor.shape

        #input_tensor (B,C,Y,X)  pooling_filter(M,N)

        # Preparation
        num_Y = (Y - self.M) // self.stride[0] +1         #the number of pooling filters in Y coordinate
        num_X = (X - self.N) // self.stride[1] +1         #the number of pooling filters in X coordinate
        self.max_position = []                            #stores the position of maximum
        output_tensor = np.zeros((B, C, num_Y, num_X))    #stores the output of maximum

        # Forward Propagation
        for b in range(B):
            for c in range(C):
                for i in range(num_Y):  # Y_coordinate of initial pooling position
                    for j in range(num_X):  # X_coordinate
                        #get the pooling region
                        o_Y = i * self.stride[0]
                        o_X = j * self.stride[1]
                        img = input_tensor[b, c, o_Y : o_Y + self.M, o_X: o_X + self.N]
                        # max pooling
                        output_tensor[b, c, i, j] = np.max(img)
                        self.max_position.append(np.argmax(img))
        self.max_position = np.array(self.max_position).reshape((B, C, num_Y, num_X))
        return output_tensor

    def backward(self, error_tensor):
        # Generalization of 1D case and calculate output error shape
        B = error_tensor.shape[0]
        C = error_tensor.shape[1]
        num_Y = error_tensor.shape[2]
        if len(error_tensor.shape) == 3:
            error_tensor = error_tensor.reshape(B, C, num_Y, 1)       # (B,C,Y',1)
            num_X =1
            out_error_shape = (B, C, self.input_shape[2])   # (B,C,Y)
        else:
            num_X = error_tensor.shape[3]
            out_error_shape = self.input_shape       # (B,C,Y,X)

        # Backward Propagation
        output_error = np.zeros(self.input_shape)    # (B,C,Y,X)
        for b in range(B):
            for c in range(C):
                for i in range(num_Y):      # Y_coordinate of initial pooling position
                    for j in range(num_X):  # X_coordinate
                        o_Y = i * self.stride[0]
                        o_X = j * self.stride[1]
                        tmp = output_error[b, c, o_Y : o_Y + self.M, o_X: o_X + self.N].reshape(-1,1)
                        tmp[self.max_position[b,c,i,j]] += error_tensor[b, c, i, j]         #error accumulation
                        output_error[b, c, o_Y : o_Y + self.M, o_X: o_X + self.N] = tmp.reshape(self.M, self.N)
        output_error = output_error.reshape(out_error_shape)
        return output_error
