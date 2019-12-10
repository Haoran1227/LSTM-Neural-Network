import numpy as np

class Sgd:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        #update weight_tensor by SGD
        weight_tensor = weight_tensor - self.learning_rate*gradient_tensor
        return weight_tensor

class SgdWithMomentum:
    def __init__(self, learninig_rate, momentum_rate):
        self.learning_rate = learninig_rate
        self.momentum_rate = momentum_rate
        self.momentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # update weight_tensor by SGD with momentum
        self.momentum = self.momentum_rate * self.momentum -self.learning_rate * gradient_tensor    #v=u*v-lr*g
        weight_tensor = weight_tensor + self.momentum       #w=w+v
        return weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.momentum = 0   #the momentum is initialized as 0
        self.penalty = 0    #the penalty is initialized as 0
        self.iteration = 1   #it stores the iteration order

    def calculate_update(self, weight_tensor, gradient_tensor):
        # update weight_tensor by Adam
        self.momentum = self.mu * self.momentum + (1 - self.mu) * gradient_tensor  # v=u*v+(1-u)*g
        self.penalty = self.rho * self.penalty + (1 - self.rho) * gradient_tensor * gradient_tensor     #r=rho*r+(1-rho)*g*g
        v_tmp = self.momentum / (1 - np.power(self.mu, self.iteration))     #v' = v/(1-mu^k)
        r_tmp = self.penalty / (1 - np.power(self.rho, self.iteration))     #r' = r/(1-rho^k)
        weight_tensor = weight_tensor - self.learning_rate * (v_tmp + np.finfo(float).eps) / (np.sqrt(r_tmp) + np.finfo(float).eps)  # w=w+v
        self.iteration += 1
        return weight_tensor
