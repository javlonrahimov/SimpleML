import numpy as np
from Layers import Base


class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.active = None

    def forward(self, input_tensor):
        self.active = 1 / (1 + np.exp(-input_tensor))
        return self.active

    def backward(self, error_tensor):
        return self.active * (1 - self.active) * error_tensor
