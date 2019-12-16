from Layers.Base import base_layer

class ReLU(base_layer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor                #save input_tensor Z_l
        input_tensor[input_tensor < 0] = 0
        return input_tensor

    def backward(self, error_tensor):
        # gradient of ReLU: when derivative<=0,it's 0;when derivative>0, it's 1
        # The derivative is respcted to Z_l, E_l = (W_l+1).T * E_l+1 * f'(Z_l)  !!!!!!!
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor
