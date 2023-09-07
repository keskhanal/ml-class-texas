from layers.layer import Layer

class ActivationLayer(Layer):
    """A class representing a layer in a neural network that applies an activation function to its input.
    Args:
        activation (function): The activation function to be applied to the input.
        activation_prime (function): The derivative of the activation function.

    Attributes:
        input (numpy.ndarray): The input to the layer.
        output (numpy.ndarray): The output of the layer after applying the activation function.

    Methods:
        forward_propagation(input_data):
            Applies the activation function to the input and returns the output.

        backward_propagation(output_error, learning_rate):
            Calculates the input error tensor using the derivative of the activation function with respect to
            the input and the output error tensor.
    """
  
    def __init__(self, activation, activation_prime) -> None:
        """
        Initializes an ActivationLayer instance
        Args:
            activation (function): The activation function to be applied to the input.
            activation_prime (function): The derivative of the activation function.
        """
        self.activation = activation
        self.activation_prime = activation_prime

    
    def forward_propagation(self, input_data):
        """
        Applies the activation function to the input and returns the output.
        Args:
            input_data (np.ndarray): The input to the layer.

        Returns:
            np.ndarray: The output of the layer after applying the activation function.
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    
    def backward_propagation(self, output_error, learning_rate):
        """
        Calculates the input error tensor using the derivative of the activation function with respect to
        the input and the output error tensor.
        Args:
            output_error (np.ndarray): The error tensor with respect to the output of the layer.
            learning_rate: The learning rate is not used because there are no learnable parameters.

        Returns:
            np.ndarray: The error tensor with respect to the input of the layer.
        """
        return self.activation_prime(self.input) * output_error
        