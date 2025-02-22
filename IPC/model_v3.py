import math
import torch
import random
import numpy as np
from torch.nn import init
from torch.nn.init import kaiming_uniform_
from features import RED, GREEN, RESET

def initialize_layer_connections(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = init.uniform_(gen_b_matrix, -bound, bound)

    return [np.array(weights), np.array(bias)]

def initialize_network_layers(network_architecture):
    parameters = []
    for size in range(len(network_architecture)-1):
        input_size = network_architecture[size]
        output_size = network_architecture[size+1]
        connections = initialize_layer_connections(input_size, output_size)
        parameters.append(connections)

    return parameters

def intermediate_activation(input_data, return_derivative=False):
    '''Use ReLU activation for intermediate layers'''
    if return_derivative:
        return np.where(input_data > 0, 1, 0)
    else:
        return np.maximum(0, input_data)

def network_output_activation(input_data):
    '''Use softmax activation for output layer'''
    # Subtract max value for numerical stability
    shifted_data = input_data - np.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = np.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    return exp_data / sum_exp_data

def forward_pass(input_data, parameters):
    activations = [input_data]
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        pre_activation = np.matmul(activation, weights)
        activation = intermediate_activation(pre_activation) if not last_layer else pre_activation
        activations.append(activation)

    return activations

def backward_pass(network_output, parameters):
    activations = []
    predicted_activation = network_output
    for each in range(len(parameters)):
        weights = parameters[each][0]
        predicted_activation = intermediate_activation(np.matmul(predicted_activation, weights))
        activations.append(predicted_activation)
    
    return activations

def calculate_activation_error(label, forward_activations, backward_activations):
    activations_errors = []
    for each in range(len(forward_activations)-1):
        # Difference between the network output and expected
        if each == 0:
            activation_error = forward_activations[-1] - label
        else:
            activation_error = forward_activations[-(each+1)] - backward_activations[each-1]

        activations_errors.append(activation_error)

    return activations_errors

def update_activations(activations, activations_error, parameters):
    for each in range(len(activations)-1):
        if each == 0:
            activations[-(each+1)] += 0.5 * -(activations_error[each])
        else:
            weights = parameters[-(each)][0].T
            activations[-(each+1)] += 0.5 * -(activations_error[each]) + intermediate_activation(activations[-(each+1)], True) * np.matmul(activations_error[each-1], weights)

def update_weights(activations, activations_error, parameters):
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]
        
        pre_activation = activations[-(each+2)]
        error = activations_error[each]

        weights -= 0.00005 * (np.matmul(pre_activation.T, error) / pre_activation.shape[0])

def update_parameters(forward_activations, label, parameters, feedback_params):
    for _ in range(5):
        
        backward_activations = backward_pass(label, feedback_params)
        activations_error = calculate_activation_error(label, forward_activations, backward_activations)

        update_activations(forward_activations, activations_error, parameters)
        update_weights(forward_activations, activations_error, parameters)

def ipc_neural_network_v3(size: list):
    parameters = initialize_network_layers(size)

    # Feedback prediction params
    feedback_parameters = initialize_network_layers([10, 256, 256])

    def train_runner(dataloader):
        for input_image, label in dataloader:
            forward_activations = forward_pass(input_image, parameters)
            update_parameters(forward_activations, label, parameters, feedback_parameters)

    def test_runner(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            neurons_activations = forward_pass(batched_image, parameters)[-1]
            batch_accuracy = (neurons_activations[-1].argmax(axis=-1) == batched_label.argmax(axis=-1)).mean()
            for each in range(len(batched_label)//10):
                model_prediction = neurons_activations[each].argmax()
                if model_prediction == batched_label[each].argmax(axis=-1): correctness.append((model_prediction.item(), batched_label[each].argmax(axis=-1).item()))
                else: wrongness.append((model_prediction.item(), batched_label[each].argmax(axis=-1).item()))
            print(f'Number of samples: {i+1}\r', end='', flush=True)
            accuracy.append(np.mean(batch_accuracy))
        random.shuffle(correctness)
        random.shuffle(wrongness)
        print(f'{GREEN}Model Correct Predictions{RESET}')
        [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 5]
        print(f'{RED}Model Wrong Predictions{RESET}')
        [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 5]
        return np.mean(np.array(accuracy)).item()

    return train_runner, test_runner
