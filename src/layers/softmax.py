import numpy as np


def softmax(x):
    num = np.exp(x - np.max(x))
    return num / num.sum(axis=1).reshape(-1, 1)
