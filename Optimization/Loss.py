import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None
        # self.loss = 0.0       #不能用self.loss，除非每次运行forward都要先把self.loss置为0

    def forward(self, input_tensor, label_tensor):
        #input_tensor shape: batch_size * num_output_layers
        #label_tensor is matrix of one-hot row vectors
        self.input_tensor = input_tensor
        tmp = np.sum(input_tensor * label_tensor + np.finfo(float).eps * label_tensor, axis=1)
        loss = -np.sum(np.log(tmp))                  # L = sum_b(-ln(y_hat_k+eps)) where y_k=1
        return loss

    def backward(self, label_tensor):
        #derivative C'(vector_y_hat) = 0 where y_k!=1; -1/y_hat_k where y_k=1
        error_tensor = -np.divide(label_tensor, self.input_tensor)          #e=-y/y_hat
        return error_tensor
