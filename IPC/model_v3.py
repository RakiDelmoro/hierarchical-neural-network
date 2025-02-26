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
    '''Use Sigmoid activation for intermediate layers'''
    if return_derivative:
        return input_data * (1 - input_data)
    else:
        return 1 / (1 + np.exp(-input_data))

def network_output_activation(input_data):
    '''Use softmax activation for output layer'''
    # Subtract max value for numerical stability
    shifted_data = input_data - np.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = np.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    return exp_data / sum_exp_data

def forward_pass_training(activations, parameters):
    predicted_activations = []
    for layer_idx in range(len(parameters)):
        weights = parameters[layer_idx][0]
        last_layer_idx = len(parameters)-1
        activation = activations[layer_idx]

        pre_activation = np.matmul(activation, weights)

        if layer_idx != last_layer_idx:
            predicted = intermediate_activation(pre_activation)
        else:
            predicted = network_output_activation(pre_activation)

        predicted_activations.append(predicted)

    return predicted_activations

def forward_pass(input_image, parameters):
    activation = input_image
    predicted_activations = []
    for layer_idx in range(len(parameters)):
        weights = parameters[layer_idx][0]
        last_layer_idx = len(parameters)-1

        pre_activation = np.matmul(activation, weights)

        if layer_idx != last_layer_idx:
            activation = intermediate_activation(pre_activation)
        else:
            activation = network_output_activation(pre_activation)

        predicted_activations.append(activation)

    return predicted_activations

def calculate_activation_error(init_activations, predicted_activations, label=None):
    activations_error = []
    for each in range(len(predicted_activations)):
        last_layer_idx = len(predicted_activations)-1
        if each == last_layer_idx:
            if label is None:
                error = init_activations[each+1] - predicted_activations[each]
            else:
                error = label - predicted_activations[each]
        else:
            error = init_activations[each+1] - predicted_activations[each]

        activations_error.append(error)

    return activations_error


def update_activations(activations, activations_error, parameters):
    # Start with 1 layer activation index since the 0 idx is the image and we don't need to update that.
    for layer_idx in range(1, len(activations)):
        last_layer_idx = len(activations)-1

        if layer_idx == last_layer_idx:
            # Derivative of Mean-Squared Error
            current_error = -(2 * activations_error[-1])
            activations[layer_idx] += (0.5 * current_error)
            activations[layer_idx] = network_output_activation(activations[layer_idx])
        else:
            weights = parameters[layer_idx][0].T
            previous_error = activations_error[layer_idx]
        
            propagated_error = np.matmul(previous_error, weights)        
            backprop_term = intermediate_activation(activations[layer_idx], return_derivative=True) * propagated_error
            # Derivative of Mean-Squared Error
            current_error = -(2 * activations_error[layer_idx-1])

            activations[layer_idx] += 0.5 * (current_error + backprop_term)
            activations[layer_idx] = intermediate_activation(activations[layer_idx])

def update_weights(activations, activations_error, parameters):
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]

        pre_activation = activations[-(each+2)]
        error = activations_error[-(each+1)]

        nudge = np.matmul(error.T, pre_activation).T
        weights += 0.0001 * (nudge / error.shape[0])

def initial_activations(network_architecture, input_image):
    activations = []
    batch_size = input_image.shape[0]
    for size in network_architecture:
        activation = np.zeros(shape=(batch_size, size), dtype=np.float32)
        activations.append(activation)
    
    # All activations are zero expect for the first which is the image
    activations[0] = input_image

    return activations

def ipc_neural_network_v3(size: list):
    parameters = initialize_network_layers(size)

    def train_runner(dataloader):
        each_batch_loss = []
        for input_image, label in dataloader:
            # Initial activations
            activations = initial_activations(size, input_image)
            # Predicted_activations
            loss = 0.0
            for _ in range(100):
                predicted_activations = forward_pass_training(activations, parameters)
                # Get the network prediction about the activations and calculate the error between the previous activations
                activations_error = calculate_activation_error(activations, predicted_activations, label)

                # Update the initial activations. Order of update -> LEFT to RIGHT
                update_activations(activations, activations_error, parameters)
                # Update the network weights to encourge to have the same value as the updated activations
                update_weights(activations, activations_error, parameters)

                # Summed total of free energy as stated in the paper sum(|x(l) - u(l)|**2)
                # This error should minimize 
                summed_error = sum([np.mean(error**2) for error in activations_error])

                loss += summed_error

            each_batch_loss.append(loss)
        
        return np.mean(np.array(each_batch_loss))

    def test_runner(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            neurons_activations = forward_pass(batched_image, size, parameters)[-1]
            batch_accuracy = (neurons_activations.argmax(axis=-1) == batched_label.argmax(axis=-1)).mean()
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
