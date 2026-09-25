import numpy as np
from neuron import Neuron
from layer import Layer
from network import Network
from activations import relu
from activations import sigmoid
from losses import squared_error

# layer1 = Layer([
#     Neuron(weights=[1, -1], bias=0),
#     Neuron(weights=[-1, 1], bias=0),
# ])
#
# layer2 = Layer([
#     Neuron(weights=[1, 1], bias=0),
# ])
#
# network = Network([
#     layer1,
#     layer2
# ])
#
# inputs = [3.0, 2.0]
#
# output = network.forward(inputs)
#
# print(output)
#
# values = [-3, -1, 0, 1, 3]
#
# for value in values:
#     print(value, relu(value))

values = [-5, -2, -1, 0, 1, 2, 5]

for value in values:
    print(value, sigmoid(value))

print(squared_error(0.8,1))
print(squared_error(0.1, 1))



