import numpy as np
from PendulumModel.utils import tanh

def neural_network(parameters):

    def train_forward(input_data):
        activation = input_data
        activations = [activation]
        for each in range(len(parameters)):
            last_layer = len(parameters)-2
            weights = parameters[each][0]
            bias = parameters[each][1]
            pre_activation = np.matmul(activation, weights) + bias
            activation = tanh(pre_activation) 
            activations.append(activation)
        return activations

    def test_forward(input_data):
        activation = input_data
        for each in range(len(parameters)):
            last_layer = len(parameters)-2
            weights = parameters[each][0]
            bias = parameters[each][1]
            pre_activation = np.matmul(activation, weights) + bias
            activation = tanh(pre_activation) 
        return activation

    return train_forward, test_forward