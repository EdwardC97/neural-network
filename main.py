import numpy as np
from neuron import Neuron
from layer import Layer
from network import Network
from activations import relu
from activations import sigmoid
from losses import squared_error
from optimiser import update_weight

inputs = [1, 2, 3, 4]
targets = [5, 10, 15, 20]

weight = 1.0
learning_rate = 0.0216

for epoch in range(20):

    for x, target in zip(inputs, targets):

        # Forward pass
        prediction = x * weight

        # Loss
        loss = (prediction - target) ** 2

        # Gradients
        loss_gradient = 2 * (prediction - target)
        prediction_gradient = x
        weight_gradient = loss_gradient * prediction_gradient

        # Update weight
        weight = weight - learning_rate * weight_gradient

    print("Epoch:", epoch, "Weight:", weight)