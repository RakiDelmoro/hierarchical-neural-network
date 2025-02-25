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

def forward_pass(activations, label, parameters):
    activations_error = []
    for layer_idx in range(len(parameters)):
        weights = parameters[layer_idx][0]
        current_activation = activations[layer_idx]

        last_layer_idx = len(parameters)-1

        if layer_idx == last_layer_idx:
            pre_activation = np.matmul(current_activation, weights)
            predicted = network_output_activation(pre_activation)
            # Output layer calculate error between the target label
            error = label - predicted 
        else:
            pre_activation = np.matmul(current_activation, weights)
            # Use ReLU activation
            predicted = intermediate_activation(pre_activation)

            # Calulate the error between the predicted activation and initial activation
            error = activations[layer_idx+1] - predicted
    
        activations_error.append(error)

    return activations_error

def update_activations(activations, activations_error, parameters):
    # Start with 1 layer activation index since the 0 idx is the image and we don't need to update that.
    for layer_idx in range(1, len(activations)):
        last_layer_idx = len(activations)-1

        if layer_idx == last_layer_idx:
            current_error = -( 2 * activations_error[-1])
            activations[layer_idx] += (0.5 * current_error) / 2098
            activations[layer_idx] = network_output_activation(activations[layer_idx])
        else:
            weights = parameters[layer_idx][0].T
            previous_error = activations_error[layer_idx]
        
            propagated_error = np.matmul(previous_error, weights)        
            derivative_activation = intermediate_activation(activations[layer_idx], return_derivative=True)
            current_error = -(2 * activations_error[layer_idx-1])

            activations[layer_idx] += 0.5 * (current_error + (derivative_activation * propagated_error)) / 2098
            activations[layer_idx] = intermediate_activation(activations[layer_idx])

def update_weights(activations, activations_error, parameters):
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]
        
        # if activations[-(each+2)].shape[1] != 784:
        #     pre_activation = intermediate_activation(activations[-(each+2)])
        # else:
        pre_activation = activations[-(each+2)]

        error = activations_error[-(each+1)]
        nudge = np.matmul(error.T, pre_activation).T
        weights += 0.0001 * (nudge / pre_activation.shape[0])

def initial_activations(network_architecture, input_image):
    activations = []
    batch_size = input_image.shape[0]
    for size in network_architecture:
        activation = np.zeros(shape=(batch_size, size), dtype=np.float32)
        activations.append(activation)
    
    # All activations are zero expect for the first which is the image
    activations[0] = input_image

    return activations

def forward_in_network(input_image, parameters):
    activation = input_image
    activations = []
    for idx in range(len(parameters)):
        weights = parameters[idx][0]
        last_layer_idx = len(parameters)-1

        if idx == last_layer_idx:
            pre_activation = np.matmul(activation, weights)
            activation = network_output_activation(pre_activation)
        else:
            pre_activation = np.matmul(activation, weights)
            activation = intermediate_activation(pre_activation)

        activations.append(activation)

    return activations

def calculate_activation_error(initial_activations, predicted_activations):
    activations_error = []

    for each in range(len(predicted_activations)):
        error = initial_activations[each+1] - predicted_activations[each]
        activations_error.append(error)

    return activations_error

def predict(input_image, parameters, size):
    activations = initial_activations(size, input_image)

    for _ in range(10):
        predicted_activations = forward_in_network(input_image, parameters)
        activations_errors = calculate_activation_error(activations, predicted_activations)

        update_activations(activations, activations_errors, parameters)

    return activations[-1]

def ipc_neural_network_v3(size: list):
    parameters = initialize_network_layers(size)

    def train_runner(dataloader):
        each_batch_loss = []
        for input_image, label in dataloader:
            init_activations = initial_activations(size, input_image)
            loss = 0.0
            for _ in range(10):
                # Get the network prediction about the activations and calculate the error between the previous activations
                activations_error = forward_pass(init_activations, label, parameters)

                # Update the initial activations. Order of update -> LEFT to RIGHT
                update_activations(init_activations, activations_error, parameters)
                # Update the network weights to encourge to have the same value as the updated activations
                update_weights(init_activations, activations_error, parameters)

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
            neurons_activations = predict(batched_image, parameters, size)
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
