import math
import torch
import random
import numpy as np
from torch.nn import init
from torch.nn.init import kaiming_uniform_
from features import RED, GREEN, RESET

def tanh(input_data, return_derivative=False):
    if return_derivative:
        input_data = (np.exp(input_data) - np.exp(-input_data))/(np.exp(input_data) + np.exp(-input_data))
        return 1 - input_data * input_data
    else:
        return (np.exp(input_data) - np.exp(-input_data))/(np.exp(input_data) + np.exp(-input_data))

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

def refine_activations(activations, activations_loss, parameters, learning_rate):
    new_activations = []
    for each in range(len(activations_loss)):
        weights = parameters[-each][0].T
        if each == 0:
            # Zero term since we don't have a previous layer in last layer
            delta_x =  (0.5 * (-activations_loss[each] + 0))
        else:
            propagated_error = np.matmul(activations_loss[each-1], weights)
            term = tanh(activations[-(each+1)], return_derivative=True) * propagated_error
            delta_x = (0.5 * (-activations_loss[each] + term))

        new_activation = activations[-(each+1)] + delta_x
        new_activations.append(new_activation)
    new_activations.append(activations[0])
    # From output to input -> input to output
    return new_activations

def backward_pass(forward_pass_output, label, parameters):
    activations = [label]
    activation = forward_pass_output
    # top to bottom
    reversed_parameters = parameters[::-1]
    for each in range(len(reversed_parameters)):
        weights = reversed_parameters[each][0]
        activation = np.matmul(activation, weights.T)
        activations.append(activation)
    return activations

def forward_pass(input_data, parameters):
    activations = [input_data]
    activation = input_data
    for each in range(len(parameters)):
        last_layer = len(parameters)-2
        weights = parameters[each][0]
        pre_activation = np.matmul(activation, weights)
        activation = tanh(pre_activation) if last_layer else pre_activation
        activations.append(activation)
    return activations

def calculate_activation_error(forward_activations, label, parameters):
    loss = np.mean((forward_activations[-1] - label)**2)
    activation_errors = [forward_activations[-1] - label]

    reversed_activations = forward_activations[::-1]
    for each in range(len(reversed_activations)-2):
        predicted_activation = tanh(np.matmul(reversed_activations[each], parameters[-(each+1)][0].T))
        error = predicted_activation - reversed_activations[each+1]
        loss += np.mean(error**2)
        activation_errors.append(error)
    return loss, activation_errors

def update_connection(activations, activations_error, parameters):
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]

        activation = activations[each+1]
        activation_error = activations_error[each]

        hebbs_rule = (0.0001 * np.matmul(activation.T, activation_error) / activation.shape[0])
        weights -= hebbs_rule

def update_parameters(forward_activations, expected, num_iterations, learning_rate, parameters):
    for i in range(num_iterations):
        loss, layers_activation_error = calculate_activation_error(forward_activations, expected, parameters)
        predicted_activations_refined = refine_activations(forward_activations, layers_activation_error, parameters, learning_rate)
        update_connection(predicted_activations_refined, layers_activation_error, parameters)
        print(f'Iters: {i} Loss: {loss}\r', end='', flush=True)

def neural_network(size: list):
    parameters = parameters_init(size)

    def train_runner(dataloader):
        losses = []
        for input_image, label in dataloader:
            forward_activations = forward_pass(input_image, parameters)
            update_parameters(forward_activations, label, 50, 0.1, parameters)
            loss = np.mean((forward_activations[-1] - label)**2)
            losses.append(loss)

        return np.mean(np.array(losses))
    
    def test_runner(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            neurons_activations = forward_pass(batched_image, parameters)
            batch_accuracy = (neurons_activations[-1].argmax(axis=-1) == batched_label.argmax(axis=-1)).mean()
            for each in range(len(batched_label)//10):
                model_prediction = neurons_activations[-1][each].argmax()
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
