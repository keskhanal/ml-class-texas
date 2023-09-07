import numpy as np
from layers.layer import Layer

class FCLayer(Layer):
    """A fully connected layer of neurons in a neural network.
    Attributes:
        weights (np.ndarray): The weights of the layer, with shape (input_size, output_size).
        bias (np.ndarray): The bias of the layer, with shape (1, output_size).
        input (np.ndarray): The input to the layer, with shape (batch_size, input_size).
        output (np.ndarray): The output of the layer, with shape (batch_size, output_size).

    Methods:
        __init__(self, input_size, output_size): Initializes the weights and bias of the layer randomly.
        forward_propagation(self, input_data): Computes the output of the layer for a given input.
        backward_propagation(self, output_error, learning_rate): Computes the input error and updates the weights and bias.
    """

    def __init__(self, input_size, output_size) -> None:
        """Initializes the weights and bias of the layer randomly.
        Args:
            input_size (int): The number of input neurons.
            output_size (int): The number of output neurons.
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5


    def forward_propagation(self, input_data):
        """Computes the output of the layer for a given input.
        Args:
            input_data (np.ndarray): The input data to the layer, with shape (batch_size, input_size).

        Returns:
            np.ndarray: The output of the layer, with shape (batch_size, output_size).
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    

    def backward_propagation(self, output_error, learning_rate):
        """Computes the input error and updates the weights and bias for a given output error.

        Args:
            output_error (np.ndarray): The error of the output of the layer, with shape (batch_size, output_size).
            learning_rate (float): The learning rate for updating the weights and bias.

        Returns:
            np.ndarray: The error of the input to the layer, with shape (batch_size, input_size).
        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        #update parameters
        self.weights -= learning_rate*weights_error
        self.bias -= learning_rate*output_error
        
        return input_error
        
