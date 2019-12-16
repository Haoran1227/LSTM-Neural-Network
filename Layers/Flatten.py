from Layers.Base import base_layer

class Flatten(base_layer):
    # input_tensor shape:(B,H,X,Y)
    # error_tensor shape:(B,HXY)
    def __init__(self):
        super().__init__()
        self.shape = None       #store the input_tensor.shape

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        input_tensor = input_tensor.reshape(self.shape[0],-1)
        return input_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.shape)
        return error_tensor
