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

def refine_activations(static_activations, predicted_activations, prediction_error, parameters, learning_rate):
    new_activations = []
    for each in range(len(predicted_activations)-1):
        weights = parameters[-(each+1)][0].T
        if each == 0:
            delta_x = 0.5 * (-prediction_error[each])
        else:
            current_error = static_activations[-(each+1)] - predicted_activations[-(each+1)]
            deriv_activation = relu(predicted_activations[-(each+1)], return_derivative=True)
            term = deriv_activation * prediction_error
            delta_x =  0.5 * ((-current_error) + term)

        prediction_error = np.matmul(prediction_error, weights)

        new_activation = softmax(predicted_activations[-(each+1)] - delta_x) if each == 0 else relu(predicted_activations[-(each+1)] - delta_x)
        new_activations.append(new_activation)
    new_activations.append(static_activations[0])
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
    return activations, activations

def calculate_error_neurons(forward_activation, backward_activation, parameters):
    activations_errors = []
    # neurons_errors = np.zeros(shape=(2098))
    for each in range(len(parameters)):
        error_activation = 2 * (forward_activation[-(each+1)] - backward_activation[each])
        activations_errors.append(error_activation)
    return activations_errors

def update_connection(static_activations, errors, parameters):
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]

        pre_synaptic = static_activations[-(each+2)]
        # post_synaptic = static_activations[-(each+1)]
        activation_error = errors[-(each+1)]

        hebbs_rule = (0.00005 * np.matmul(pre_synaptic.T, activation_error)) / pre_synaptic.shape[0]
        weights += hebbs_rule

def inference_step(network_activations, errors, parameters):
    refined_activations = []
    for each in range(len(errors)):
        if each == 0:
            refined_activation = network_activations[-(each+1)] - 0.5 * (-errors[each])
        else:
            propagated_error = np.matmul(errors[each-1], parameters[-(each)][0].T)
            deriv_activation = relu(network_activations[-(each+1)], return_derivative=True)
            refined_activation = network_activations[-(each+1)] - 0.5 * ((-errors[each]) + (deriv_activation * propagated_error))

        refined_activations.append(refined_activation)
    refined_activations.append(network_activations[0])
    return refined_activations[::-1]

def calculate_activation_dissimilarity(forward_activations, refined_activations):
    dissimilarities = []
    for each in range(len(refined_activations)):
        f_activation = forward_activations[each]
        r_activation = refined_activations[each]
        disimilarity = f_activation - r_activation
        dissimilarities.append(disimilarity)
    return dissimilarities

def update_parameters(forward_activations, label, parameters, learning_rate):
    # Num iteration to refine internal activations
    for _ in range(5):
        activations_errors = backward_pass(forward_activations, label, parameters)
        # Refined network activations
        refined_activations = inference_step(forward_activations, activations_errors, parameters)
        # x(l) - u(l)
        dissimilarities = calculate_activation_dissimilarity(refined_activations, refined_activations)
        update_connection(forward_activations, dissimilarities, parameters)
        forward_activations = refined_activations

def backward_pass(network_activations, label, parameters):
    activations_errors = []
    error = network_activations[-1] - label
    activations_errors.append(error)

    for each in range(len(parameters)-1):
        weights = parameters[-(each+1)][0].T
        pre_activity_error = np.matmul(error, weights)
        error = pre_activity_error * relu(network_activations[-(each+2)], return_derivative=True)
        activations_errors.append(error)
    return activations_errors

def ipc_neural_network(size: list):
    parameters = parameters_init(size)

    def train_runner(dataloader):
        losses = []
        for input_image, label in dataloader:
            # Up-Bottom Predictions
            activations_guess, activations_refined = forward_pass(input_image, parameters)
            update_parameters(activations_guess, label, parameters, 0.01)
            losses.append(1.0)

        return np.mean(np.array(losses))

    def test_runner(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            neurons_activations = forward_pass(batched_image, parameters)[0]
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
