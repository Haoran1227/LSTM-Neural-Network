import numpy as np

class base_layer:
    def __init__(self):
        self.phase = 'train'
        # self.weights is defined for not-trainable layers
        # 1. we can know whether the layer is trainable by checking whether weights is equal to zero.
        # 2. if it's trainable, we can employ "layer.optimizer.regularizer.norm(weights)" to calculate norm loss
        self.weights = None


