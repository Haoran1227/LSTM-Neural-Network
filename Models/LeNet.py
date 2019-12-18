from NeuralNetwork import NeuralNetwork
from Layers import *
from Optimization import *
import math

def build():
    lr = math.pow(5,-4)
    alpha = 4e-4
    optimizer = Optimizers.Adam(lr)
    optimizer.add_regularizer(Constraints.L2_Regularizer(alpha))
    net = NeuralNetwork(optimizer,
                        Initializers.He(),
                        Initializers.Constant(0.1))

    net.loss_layer = Loss.CrossEntropyLoss()

    cl_1 = Conv.Conv((1,1), (1,5,5), 6)
    net.append_trainable_layer(cl_1)
    net.layers.append(ReLU.ReLU())

    pl_2 = Pooling.Pooling((2,2), (2,2))
    net.layers.append(pl_2)

    cl_3 = Conv.Conv((1,1), (1,5,5), 16)
    net.append_trainable_layer(cl_3)
    net.layers.append(ReLU.ReLU())

    pl_4 = Pooling.Pooling((2,2), (2,2))    #16*7*7
    net.layers.append(pl_4)

    cl_5 = Conv.Conv((100,100), (1,15,15), 120)       #stride and convolution shape large enough, it's the same use as valid convolution
    net.append_trainable_layer(cl_5)
    net.layers.append(ReLU.ReLU())

    net.layers.append(Flatten.Flatten())    #120

    fcl_1 = FullyConnected.FullyConnected(120, 84)
    net.append_trainable_layer(fcl_1)
    net.layers.append(ReLU.ReLU())

    fcl_2 = FullyConnected.FullyConnected(84, 10)
    net.append_trainable_layer(fcl_2)
    net.layers.append(SoftMax.SoftMax())

    return net