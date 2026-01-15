# This file contains the definitions for the LossFunction class and various common loss functions

import numpy as np

class LossFunction:

    def __init__(self, signature, derivative_signature):
        self.signature = signature
        self.derivative_signature = derivative_signature



# Let's also define the loss functions
# Binary Cross Entropy Loss function
bce_loss = LossFunction(
    signature=lambda target, output: -np.mean(
        target * np.log(output + 1e-15) + (1 - target) * np.log(1 - output + 1e-15)
    ),

    # We can also pair this with the sigmoid function to have another elegant loss function
    derivative_signature=lambda target, output: (
        (output - target) / (output * (1 - output) + 1e-15)
    )
)


# Categorical Cross-Entropy Loss Function
cce_loss = LossFunction(
    signature=lambda target, output: -np.sum(
        target * np.log(output + 1e-15)
    ),
    derivative_signature=lambda target, output: (
        -(target / (output + 1e-15))
    )
)