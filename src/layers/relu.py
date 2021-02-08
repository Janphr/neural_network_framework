from .layer import Layer


class ReLu(Layer):
    def forward(self, x):
        mask = x > 0
        return x * mask, lambda delta: delta * mask
