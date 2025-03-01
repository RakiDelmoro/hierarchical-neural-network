import math
import torch
import random
import numpy as np
from torch.nn import init
from torch.nn.init import kaiming_uniform_
from features import RED, GREEN, RESET

def cross_entropy(expected, model_prediction):
    epsilon = 1e-10  # Small value to prevent log(0)
    loss = -np.sum(expected * np.log(model_prediction + epsilon), axis=1)

    return loss

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

def forward_pass(activations, parameters):
    predicted_activations = []
    for layer_idx in range(len(parameters)):
        weights = parameters[layer_idx][0]
        last_layer_idx = len(parameters)-1
        activation = activations[layer_idx]

        pre_activation = np.matmul(activation, weights)

        if layer_idx != last_layer_idx:
            predicted = intermediate_activation(pre_activation)
        else:
            # predicted = pre_activation
            predicted = network_output_activation(pre_activation)

        predicted_activations.append(predicted)

    return predicted_activations

def calculate_activation_error(activations, predicted_activations, expected_output=None):
    activations_error = []
    for each in range(len(predicted_activations)):
        last_layer_idx = len(predicted_activations)-1
        if each == last_layer_idx:
            if expected_output is None:
                error = activations[each+1] - predicted_activations[each]
            else:
                error = expected_output - predicted_activations[each]
        else:
            error = activations[each+1] - predicted_activations[each]

        activations_error.append(error)
    
    return activations_error

def update_activations(activations, activations_error, parameters):
    # Start with 1 layer activation index since the 0 idx is the image and we don't need to update that.
    for layer_idx in range(len(activations_error)):
        last_layer_idx = len(activations_error)-1

        if layer_idx == last_layer_idx:
            current_error = -activations_error[-1]

            activations[-1] += (0.5 * current_error)

        else:
            weights = parameters[layer_idx+1][0].T
            previous_error = activations_error[layer_idx+1]

            propagated_error = np.matmul(previous_error, weights)        
            backprop_term = intermediate_activation(activations[layer_idx+1], return_derivative=True) * propagated_error
            current_error = -activations_error[layer_idx]

            activations[layer_idx+1] += 0.5 * (current_error + backprop_term)

def update_weights(activations, activations_error, parameters):
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]

        # if each == len(parameters)-1:
        pre_activation = activations[-(each+2)]
        # else:
            # pre_activation = intermediate_activation(activations[-(each+2)])

        error = activations_error[-(each+1)]

        nudge = np.matmul(error.T, pre_activation).T
        weights += 0.00005 * (nudge / error.shape[0])

def initial_activations(network_architecture, input_image, label=None):
    activations = []
    batch_size = input_image.shape[0]
    for size in network_architecture:
        activation = np.zeros(shape=(batch_size, size), dtype=np.float32)
        activations.append(activation)

    # All initialize activations are zero expect for the first index which is our input image
    activations[0] = input_image

    if label is not None:
        activations[-1] = label

    return activations

def predict(input_image, parameters, size):
    activations = initial_activations(size, input_image)
    for _ in range(150):
        predicted_activations = forward_pass(activations, parameters)
        activations_error = calculate_activation_error(activations, predicted_activations)

        update_activations(activations, activations_error, parameters)

    return predicted_activations[-1]

def ipc_neural_network_v3(size: list):
    parameters = initialize_network_layers(size)

    def train_runner(dataloader):
        each_batch_loss = []
        for input_image, label in dataloader:
            # Initial activations
            activations = initial_activations(size, input_image)
            # [h1, h2, output]
            # Predicted_activations
            losses = []
            for _ in range(150):
                predicted_activations = forward_pass(activations, parameters)
                # Get the network prediction about the activations and calculate the error between the previous activations
                activations_error = calculate_activation_error(activations, predicted_activations, label)

                # Inference and Learning Phase
                update_activations(activations, activations_error, parameters)
                update_weights(activations, activations_error, parameters)

                loss = cross_entropy(label, predicted_activations[-1])
                losses.append(np.mean(loss))

            each_batch_loss.append(sum(losses))

        return np.mean(np.array(each_batch_loss))

    def test_runner(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            model_output = predict(batched_image, parameters, size)
            batch_accuracy = (model_output.argmax(axis=-1) == batched_label.argmax(axis=-1)).mean()
            for each in range(len(batched_label)//10):
                model_prediction = model_output[each].argmax()
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
