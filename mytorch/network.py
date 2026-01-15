# This file contains the definition for the NeuralNetwork class

import numpy as np

from .layer import *

from .activation import *

from .loss import * 

# [neurons in input layer, number of hidden layers and neurons in each hidden layer, activation function of the hidden layer, neurons in output layer, activation function of the output layer]
# A neural network would be made like this:
# NeuralNetwork(10, [4 4], relu, 4, sigmoid)
# This means the network has 10 input neurons, then 2 hidden layers with 4 neurons in each layer, and 4 output neurons
class NeuralNetwork:

    def __init__(self, layers, loss_function):

        # Since we will have all the required information about the various layers of the network when creating layers
        # we can just set the network and loss function to those values and we are done
        self.network = layers
        self.loss_function = loss_function

    
    def __repr__(self):
        return f"Network = [{self.network}]"


    # Let's define the function that actually feeds the input forward to the network to produce the output
    def feed_forward(self, input):

        for i in range(len(self.network)):

            output = self.network[i].output(input)

            input = output

        return output


    def _forward_pass(self, input, target):

        output = self.feed_forward(input)

        self.loss = self.loss_function.signature(target, output)

        # CHECK: If using Softmax + CCE, the derivative is just (Output - Target)
        # This is numerically stable and avoids the 'increasing loss' issue.
        if self.network[-1].activation_function == softmax and self.loss_function == cce_loss:
            self.delta = output - target
        else:
            # Standard chain rule for other combinations (like Sigmoid/BCE)
            final_layer = self.network[-1]
            self.delta = (
                self.loss_function.derivative_signature(target, final_layer.A) *
                final_layer.activation_function.derivative_signature(final_layer.Z)
            )

        return output

    
    def _backpropagate(self, learning_rate):

        for i in reversed(range(len(self.network))):

            # In forward pass, we have the delta for the final layer, so we just change it till we reach the input
            current_delta = self.delta

            layer = self.network[i]

            # For every layer (coming from the back), we find the gradients for that layer as
            gradW = np.dot(current_delta, layer.inputA.T)
            gradB = current_delta

            # While we haven't reached the input layer, we need to change the current delta
            if i > 0:

                previous_layer = self.network[i-1]

                # Propagate the error to the next layer
                self.delta = np.dot(layer.W.T, current_delta) * previous_layer.activation_function.derivative_signature(previous_layer.Z)
            
            layer.W -= learning_rate * gradW
            layer.B -= learning_rate * gradB


    # The forward_pass and the backward_pass are the internal functions for training
    # Now, we actually define the train function exposed via the API
    def train(self, input, target, learning_rate):

        # First, perform forward pass
        self._forward_pass(input, target)

        # Now, perform backward pass with the learning rate
        self._backpropagate(learning_rate)

        # Now, we return the loss produced by the forward pass
        return self.loss
