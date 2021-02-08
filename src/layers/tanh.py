from .layer import Layer
import numpy as np


class Tanh(Layer):
    def forward(self, x):

        def backward(delta):
            return 1 - np.square(np.tanh(delta))

        return np.tanh(x), backward
