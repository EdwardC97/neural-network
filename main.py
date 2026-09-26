import numpy as np
from neuron import Neuron
from layer import Layer
from network import Network
from activations import relu
from activations import sigmoid
from losses import squared_error
from optimiser import update_weight

# inputs = [1, 2, 3, 4]
# targets = [5, 10, 15, 20]
#
# weight = 1.0
# learning_rate = 0.0216
#
# for epoch in range(20):
#
#     for x, target in zip(inputs, targets):
#
#         # Forward pass
#         prediction = x * weight
#
#         # Loss
#         loss = (prediction - target) ** 2
#
#         # Gradients
#         loss_gradient = 2 * (prediction - target)
#         prediction_gradient = x
#         weight_gradient = loss_gradient * prediction_gradient
#
#         # Update weight
#         weight = weight - learning_rate * weight_gradient
#
#     print("Epoch:", epoch, "Weight:", weight)


weight = 3
bias = 2

learning_rate = 0.1

inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
targets = [7, 14, 21, 28, 35, 42, 49, 56, 64, 70]

for epoch in range (30):

    total_weight_gradient = 0
    total_bias_gradient = 0
    total_loss = 0

    for x, target in zip(inputs, targets):

        prediction = x * weight + bias

        loss = (prediction - target)**2
        total_loss += loss

        loss_gradient = 2 * (prediction - target)
        prediction_gradient = x
        weight_gradient = loss_gradient * prediction_gradient
        bias_gradient = loss_gradient

        total_weight_gradient += weight_gradient
        total_bias_gradient += bias_gradient

    weight = weight - learning_rate * total_weight_gradient
    bias = bias - learning_rate + total_bias_gradient

    print("Epoch:", epoch,
              "Loss:", total_loss,
              "Weight:", weight,
              "Bias:", bias)