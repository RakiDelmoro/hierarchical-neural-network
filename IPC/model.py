import math
import torch
import numpy as np
from torch.nn import init
from torch.nn.init import kaiming_uniform_

def initializer(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = init.uniform_(gen_b_matrix, -bound, bound)
    return [np.array(weights), np.array(bias)]

def parameters_init(network_architecture: list):
    parameters = []
    for size in range(len(network_architecture)-1):
        input_size = network_architecture[size]
        output_size = network_architecture[size+1]
        connections = initializer(input_size, output_size)
        parameters.append(connections)
    return parameters

def calculate_activation_error(forward_activations, backward_activations):
    # Mean squared error for each layer activation
    error = 0
    layers_neurons_error = []
    for each in range(len(forward_activations)-1):
        error += np.mean((forward_activations[each+1] - backward_activations[-(each+1)])**2)
        # Derivative of mean squared error 
        layer_error = 2 * (forward_activations[each+1] - backward_activations[-(each+1)])
        layers_neurons_error.append(layer_error)
    return error, layers_neurons_error

def refine_activation(activations, activations_loss, learning_rate):
    new_activations = []
    for each in range(len(activations)):
        new_activation = activations[each] - (learning_rate * activations_loss[-(each+1)])
        new_activations.append(new_activation)
    return new_activations

def backward_pass(forward_pass_output, label, parameters):
    activations = [label]
    activation = forward_pass_output
    # top to bottom
    reversed_parameters = parameters[::-1]
    for each in range(len(reversed_parameters)-1):
        weights = reversed_parameters[each][0]
        bias = reversed_parameters[each][1]
        activation = np.matmul(activation, weights.T) + bias
        activations.append(activation)
    return activations

def forward_pass(input_data, parameters):
    activations = [input_data]
    activation = input_data
    for each in range(len(parameters)):
        weights = parameters[each][0]
        bias = parameters[each][1]
        activation = np.matmul(activation, weights) + bias
        activations.append(activation)
    return activations

def update_connection(activations):
    # TODO: Update the connection weights
    pass

def update_parameters(forward_activations, predicted_activations, num_iterations, learning_rate, parameters):
    for _ in range(num_iterations):
        error, layers_activation_error = calculate_activation_error(forward_activations, predicted_activations)
        refine_activations = refine_activation(predicted_activations, layers_activation_error, learning_rate)
        update_connection(refine_activations)

def neural_network(size: list):
    parameters = parameters_init(size)

    def train_runner(dataloader):
        for input_image, label in dataloader:
            forward_activations = forward_pass(input_image, parameters)
            predicted_activations = backward_pass(forward_activations[-1], label, parameters)
            update_parameters(forward_activations, predicted_activations, 10, 0.1, parameters)

    return train_runner

input_x = [(np.random.rand(1, 3), np.random.rand(1, 1))] * 10
network = neural_network([3, 5, 1])
print(network(input_x)[-1])
