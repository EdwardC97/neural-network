import numpy as np
from neuron import Neuron
from layer import Layer
from network import Network

layer1 = Layer([
    Neuron(weights=[1, -1], bias=0),
    Neuron(weights=[-1, 1], bias=0),
])

layer2 = Layer([
    Neuron(weights=[1, 1], bias=0),
])

network = Network([
    layer1,
    layer2
])

inputs = [3.0, 2.0]

output = network.forward(inputs)

print(output)




