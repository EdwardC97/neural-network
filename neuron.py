import numpy as np
from activations import step

class Neuron:

    #constructor
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    #method called forward which accepts parameter called inputs, returns the neuron calculation
    # def forward(self, inputs):
    #     weighted_sum = np.dot(inputs, self.weights) + self.bias
    #     return step(weighted_sum)

    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        print(f"Weighted sum: {weighted_sum}")

        return step(weighted_sum)





