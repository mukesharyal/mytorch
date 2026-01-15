# This file contains the definitions for the ActivationFunction class and various common activation functions

import numpy as np

# We need to define the activation function, but we will also need the derivatives of them, se we will 
# have to define a class for the activation function itself
class ActivationFunction:

    def __init__(self, signature, derivative_signature):
        self.signature = signature
        self.derivative_signature = derivative_signature


# Let's define the activation functions
# The Sigmoid function
sigmoid = ActivationFunction(
    signature = lambda x: 1 / (1 + np.exp(-x)),
    derivative_signature=lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
)

# The ReLU function
relu = ActivationFunction(
    signature = lambda x: np.maximum(0, x),
    derivative_signature = lambda x: (x > 0).astype(float)
)

# The SoftMax function
softmax = ActivationFunction(
    signature=lambda x: (
        np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)
    ),
    
    # We won't actually use the derivative for the loss as we will pair it with the CE Loss function for an 
    # elegant expression for the loss
    derivative_signature=lambda x: (
        (np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)) * (1 - (np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)))
    )
)