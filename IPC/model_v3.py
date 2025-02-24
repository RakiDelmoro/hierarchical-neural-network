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

def forward_pass(activations, parameters):
    predicted_activations = []
    activations_error = []
    for each in range(len(parameters)):
        weights = parameters[each][0]
        current_activation = activations[each+1]

        if each != 0:
            pre_activation = np.matmul(current_activation, weights.T)
            predicted = intermediate_activation(pre_activation)
        else:
            pre_activation = np.matmul(current_activation, weights.T)
            predicted = network_output_activation(pre_activation)

        error = predicted - activations[each]       
        
        predicted_activations.append(predicted)
        activations_error.append(error)

    return predicted_activations, activations_error

def update_activations(activations, activations_error, parameters):
    for each in range(1, len(activations)-1):
        weights = parameters[each-1][0]
        previous_error = activations_error[each-1]
        
        propagated_error = np.matmul(previous_error, weights)        
        derivative_activation = intermediate_activation(activations[each], return_derivative=True)
        current_error = activations_error[each]

        activations[each] -= 0.9 * ((-current_error) + (derivative_activation * propagated_error))

def update_weights(activations, activations_error, parameters):
    for each in range(len(parameters)):
        weights = parameters[each][0]
        
        pre_activation = activations[each+1]
        error = activations_error[each]

        weights -= 0.0001 * (np.matmul(error.T, pre_activation) / pre_activation.shape[0])

def initial_activations(network_architecture, input_image, expected_output=None):
    activations = []
    batch_size = input_image.shape[0]
    for size in network_architecture:
        activation = np.zeros(shape=(batch_size, size), dtype=np.float32)
        activations.append(activation)
    
    if expected_output is not None:
        activations[0] = expected_output 

    activations[-1] = input_image

    return activations

def predict(input_image, parameters, size):
    activations = initial_activations(size, input_image)
    for _ in range(10):
        predicted, activations_error = forward_pass(activations, parameters)
        update_activations(activations, activations_error, parameters)

    return predicted[0]

def ipc_neural_network_v3(size: list):
    parameters = initialize_network_layers(size)

    # Feedback prediction params
    feedback_parameters = initialize_network_layers([10, 256, 256])

    def train_runner(dataloader):
        for input_image, label in dataloader:
            initial_neurons_activations = initial_activations(size, input_image, label)
            # Num iteration 
            for _ in range(10):
                predicted_activations, activations_error = forward_pass(initial_neurons_activations, parameters)

                update_activations(initial_neurons_activations, activations_error, parameters)
                update_weights(initial_neurons_activations, activations_error, parameters)

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
