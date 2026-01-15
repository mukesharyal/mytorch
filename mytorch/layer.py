# This file contains the definition for the Layer class

import numpy as np

# Let's define a layer of neurons
class Layer:

    # In the new scheme to allow for vectorization, we will have the layer as the smallest unit of the network
    def __init__(self, num_neurons, num_inputs, activation_function):

        # Now, we will have an array with num_neurons number of neurons, but instead of neurons we will directly have
        # the weights and biases in a matrix
        # Instead of using a for loop, we can use numpy to initialise all the random values at once
        # This function will generate a matrix of dimensions [num_neurons] X [num_inputs]
        # We are using the randn for the values to be taken from a normal distribution with mean 0 and sigma 1
        # Also, we are normalizing the values so that they are not very large
        self.W = np.random.randn(num_neurons, num_inputs) * np.sqrt(1.0 / num_inputs)
        self.B = np.zeros((num_neurons, 1))
        self.activation_function = activation_function


    def __repr__(self):
        return f"Layer W = {self.W.shape}, Layer B = {self.B.shape}\n"
    

    def output(self, input):

        # Let's store the input to the layer as we might need it
        self.inputA = input

        # Let's also store the thereby formed z
        self.Z = np.dot(self.W, input) + self.B

        # Let's also store the output value produced by the layer as well
        self.A = self.activation_function.signature(self.Z)

        return self.A


    



