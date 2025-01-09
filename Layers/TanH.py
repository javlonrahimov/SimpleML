import numpy as np
from Layers import Base


class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.active = None

    def forward(self, input_tensor):
        self.active = np.tanh(input_tensor)
        return self.active

    def backward(self, error_tensor):
        return (1 - np.square(self.active)) * error_tensor
