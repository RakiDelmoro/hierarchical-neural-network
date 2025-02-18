import math
import torch
import random
import numpy as np
from torch.nn import init
from torch.nn.init import kaiming_uniform_
from features import RED, GREEN, RESET

def relu(input_data, return_derivative=False):
    if return_derivative:
        return np.where(input_data > 0, 1, 0)
    else:
        return np.maximum(0, input_data)

def softmax(input_data):
    # Subtract max value for numerical stability
    shifted_data = input_data - np.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = np.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    return exp_data / sum_exp_data

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
        # Actually we don't need to refine output activation since this activation will change or closer to expected if we correctly update the weights
        # Connecting H2 to Output
        if each == 0:
            delta_x = 0.01 * (-activations_loss[each])
        else:
            weights = parameters[-(each)][0].T
            propagated_error = np.matmul(activations_loss[each-1], weights)
            deriv_activation = relu(activations[-(each+1)], return_derivative=True)
            term = deriv_activation * propagated_error
            delta_x =  0.5 * ((-activations_loss[each]) + term)

        new_activation = activations[-(each+1)] + delta_x 

        new_activations.append(new_activation)
    new_activations.append(activations[0])
    return new_activations[::-1]

def forward_pass(input_data, parameters):
    activations = [input_data]
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        pre_activation = np.matmul(activation, weights)
        activation = relu(pre_activation) if not last_layer else softmax(pre_activation)
        activations.append(activation)
    return activations

def backward_pass(label, parameters):
    activations = [label]
    activation = label
    for each in range(len(parameters)-1):
        transposed_weights = parameters[-(each+1)][0].T                                                                                                                                                                                                                                                                       
        activation = relu(np.matmul(activation, transposed_weights))
        activations.append(activation)
    return activations

def calculate_activations_errors(forward_activations, backward_activations):
    loss = 0.0
    activations_errors = []
    for each in range(len(backward_activations)):
        predicted_activation = forward_activations[-(each+1)]
        actual_activation = backward_activations[each]
        activation_error = predicted_activation - actual_activation
        activations_errors.append(activation_error)
        # Enegy for a given layer
        loss += np.mean((activation_error**2))
    return loss, activations_errors

def update_connection(activations, activations_error, parameters):
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]

        activation = activations[-(each+2)]        
        activation_error = activations_error[each]

        hebbs_rule = (0.0001 * np.matmul(activation.T, activation_error) / activation.shape[0])
        weights -= hebbs_rule

def update_parameters(forward_activations, layers_activation_error, learning_rate, parameters):
    for _ in range(20):
        predicted_activations_refined = refine_activations(forward_activations, layers_activation_error, parameters, learning_rate)
        update_connection(predicted_activations_refined, layers_activation_error, parameters)

def neural_network(size: list):
    parameters = parameters_init(size)

    def train_runner(dataloader):
        losses = []
        for input_image, label in dataloader:
            forward_activations = forward_pass(input_image, parameters)
            backward_activations = backward_pass(label, parameters)
            total_activation_error, activations_errors = calculate_activations_errors(forward_activations, backward_activations)
            update_parameters(forward_activations, activations_errors, 0.1, parameters)
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


# This Training run only achieve 85% accuracy suggest that the IPC is not implemented correctly.

