import copy
import pickle

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []                      # stores loss in each iteration
        self.layers = []                    # stores the hidden layers
        self.data_layer = None              # stores the input layer (input data structure)
        self.loss_layer = None              # stores the loss calculation layer
        self.label_tensor = None            #Data load function must be operated one time in one training iteration
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.forward()         # see IrisData class in Helpers.py
        regularization_loss = 0         # record regularization loss
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)                      #forward propagation
            if layer.weights is not None:      # it's a trainable layer
                if layer.optimizer.regularizer is not None:     # it has regularizer
                    regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
        loss = self.loss_layer.forward(input_tensor, self.label_tensor) + regularization_loss    # use the output of network to do optimization
        return loss                                                         #the consequence is loss of network

    def backward(self,label_tensor):
        #input_tensor, self.label_tensor = self.data_layer.forward()
        #Do not operate this function again!! Because the data is shuffled, and loaded data in two times are different,
        #so if you load again, the label_tensor and input_tensor will not match!

        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):                                 #back propagation
            error_tensor = layer.backward(error_tensor)                     #update is in FullyConnected class
        pass

    def append_trainable_layer(self,layer):               #in this case, the trainable layer is fullyconnected layer or convolutional layer
        # Because the optimizer will store temporal variable(such as momentum) for each layer, so you must make a copy
        optimizer = copy.deepcopy(self.optimizer)
        layer.optimizer = optimizer
        layer.initialize(self.weights_initializer, self.bias_initializer)       #initialize weights and bias
        self.layers.append(layer)

    def train(self,iterations):
        self.phase = "train"
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward(self.label_tensor)
            print("current step:", i)

    def test(self,input_tensor):
        self.phase = "test"
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)              # forward propagation
        return input_tensor

# phase property to set "train" or "test" for each layer
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, condition):
        for layer in self.layers:
            layer.phase = condition

    def __getstate__(self):
        #return state values to be pickled.
        # data_layer, label_tensor, phase not pickled. Because they are not characteristic of reconstruction of net
        state = (self.optimizer, self.layers, self.loss_layer, self.weights_initializer, self. bias_initializer)
        return state

    def __setstate__(self, state):
        self.optimizer, self.layers, self.loss_layer, self.weights_initializer, self.bias_initializer = state
        # dropped member in __getstae__() should be also initialized, it seems like initialization function
        self.loss = []
        self.data_layer = None
        self.label_tensor = None
        self._phase = None

# functions used to save and load files
def save(filename, net):
    with open(filename, 'wb') as f:             #must be "wb", not "w"
        pickle.dump(net, f)

def load(filename, data_layer):
    with open(filename, 'rb') as f:
        net = pickle.load(f)
        net.data_layer = data_layer
    return net