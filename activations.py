import math

def step(x):
    if x >= 0:
        return 1
    return 0

#ReLU activation
def relu(x):
    return max(0, x)

#sigmoid activation
def sigmoid(x):
    return 1/ (1 + math.exp(-x))

