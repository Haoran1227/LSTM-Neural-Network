import numpy as np
class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha          #regularization parameter

    def calculate_gradient(self, weights):
        grad = self.alpha * np.sign(weights)
        return grad

    def norm(self, weights):
        a = np.sum(np.absolute(weights))
        loss = self.alpha *  a        #||a||_1 = sum(|a_i|)
        return loss

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        #grad = 2 * self.alpha * weights        # mathematical form
        grad = self.alpha * weights             # engineering form
        return grad

    def norm(self, weights):
        a=np.sum(np.square(weights))
        #loss = self.alpha * np.sum(np.square(weights))            #||a||_2 = sqrt(sum(a_i^2))
        loss = self.alpha * np.sqrt(np.sum(np.square(weights)))    # engineering form
        return loss


