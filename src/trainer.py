import numpy as np


class Trainer():
    def __init__(self, network, loss, optimizer):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

    def train_batch(self, x, target):
        # go through all layers in forward pass of the network
        y, backward = self.network.forward(x)
        # take the prediction (y) and evaluate with the target
        loss, delta = self.loss(y, target)
        # take the gradient of the evaluation (delta)
        # and go through all backward functions to update the weights and biases
        backward(delta)
        # go through the tensor references of the layers in the network and update the weights and biases
        self.network.m(self.optimizer)
        return loss

    def train(self, x, target, epochs, batch_size):
        losses = []

        for epoch in range(epochs):
            # creates an array of indices of input length and shuffles them.
            # this way each batch in each epoch is different
            p = np.random.permutation(len(x))
            loss = 0
            for i in range(0, len(x), batch_size):
                # get the random input values and their targets and send them through the network
                x_batch = x[p[i: i + batch_size]]
                target_batch = target[p[i: i + batch_size]]
                loss += self.train_batch(x_batch, target_batch)
            # since loss is the sum of all batches
            loss = loss * (batch_size / len(x))
            losses.append(loss)
            print('Epoch ' + str(epoch + 1) + ' loss: ' + str(round(loss, 4)))
        return losses
