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
    weights = kaiming_uniform_(gen_w_matrix, mode='fan_in', nonlinearity='leaky_relu')

    # Initialize biases to zero
    init.zeros_(gen_b_matrix)

    return [np.array(weights), np.array(gen_b_matrix)]

def initialize_network_layers(network_architecture):
    parameters = []
    for size in range(len(network_architecture)-1):
        input_size = network_architecture[size]
        output_size = network_architecture[size+1]
        connections = initialize_layer_connections(input_size, output_size)
        parameters.append(connections)

    return parameters

def relu(input_data, return_derivative=False):
    if return_derivative:
        x = np.maximum(0, input_data)
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, input_data)
    
def leaky_relu(input_data, return_derivative=False):
    if return_derivative:
        x = np.maximum(input_data * 0.05, input_data)
        return np.where(x > 0, 1, 0.05 * x)
    else:
        return np.maximum(input_data * 0.05, input_data)

def softmax(input_data):
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
        bias = parameters[layer_idx][1]
        last_layer_idx = len(parameters)-1
        activation = activations[layer_idx]

        pre_activation = np.matmul(activation, weights) + bias

        if layer_idx != last_layer_idx:
            predicted = leaky_relu(pre_activation)
        else:
            predicted = softmax(pre_activation)

        predicted_activations.append(predicted)

    return predicted_activations

def calculate_activation_error(current_state, predicted):
    activations_error = []
    for each in range(len(predicted)):
        error = current_state[each+1] - predicted[each]
        activations_error.append(error)

    return activations_error

def update_activations(activations, activations_error, parameters, lr):

    for layer_idx in range(len(activations_error)-1):
        weights = parameters[-(layer_idx+1)][0].T
        previous_error = activations_error[-(layer_idx+1)]

        propagated_error = np.matmul(previous_error, weights)        
        backprop_term = leaky_relu(activations[-(layer_idx+2)], return_derivative=True) * propagated_error
        current_error = -(activations_error[-(layer_idx+2)])

        activations[-(layer_idx+2)] += lr * (current_error + backprop_term)

def update_weights(activations, activations_error, parameters, m, v, lr, t):
    # TODO: Make this function more readable
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 1e-5

    t += 1
    for each in range(len(parameters)):
        weights = parameters[-(each+1)][0]
        bias = parameters[-(each+1)][1]

        pre_activation = activations[-(each+2)]
        error = activations_error[-(each+1)]

        nudge = np.matmul(pre_activation.T, error)
        grad_weights = nudge / error.shape[0]
        grad_bias = np.sum(error, axis=0) / error.shape[0]

        m_weights, m_bias = m[-(each+1)]
        v_weights, v_bias = v[-(each+1)]

        m_weights = beta1 * m_weights + (1 - beta1) * grad_weights
        v_weights = beta2 * v_weights + (1 - beta2) * (grad_weights ** 2)

        m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
        v_bias = beta2 * v_bias + (1 - beta2) * (grad_bias ** 2)

        m[-(each+1)] = (m_weights, m_bias)
        v[-(each+1)] = (v_weights, v_bias)

        # Bias correction
        m_hat_weights = m_weights / (1 - beta1**t)
        v_hat_weights = v_weights / (1 - beta2**t)
        m_hat_bias = m_bias / (1 - beta1**t)
        v_hat_bias = v_bias / (1 - beta2**t)

        # Update weights with AdamW (include weight decay)
        weights += lr * (m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)) #+ weight_decay * weights)
        weights += lr * (weight_decay * weights)
 
        # Update bias with Adam (no weight decay)
        bias += lr * (m_hat_bias / (np.sqrt(v_hat_bias) + epsilon))

def initialize_moments(size):
    # TODO: make this function more readable
    m = []
    v = []
    for each in range(len(size)-1):
        w = np.zeros(shape=(size[each], size[each+1]))
        b = np.zeros(shape=(size[each+1]))

        m.append([w, b])
        v.append([w, b])

    return m, v

def initial_activations(parameters, input_image, label=None):
    activations = [input_image]
    activation = input_image
    for each in range(len(parameters)):
        weights = parameters[each][0]
        bias = parameters[each][1]

        pre_activation = np.matmul(activation, weights) + bias
        activation = leaky_relu(pre_activation)
        activations.append(activation)

    if label is not None:
        activations[-1] = label

    return activations

def predict(input_image, parameters):
    activation = input_image
    for each in range(len(parameters)):
        weights = parameters[each][0]
        bias = parameters[each][1]

        pre_activation = np.matmul(activation, weights) + bias
        if each != len(parameters)-1:
            activation = leaky_relu(pre_activation)
        else:
            activation = softmax(pre_activation)

    return activation

def ipc_neural_network_v3(size: list, parameters_lr):
    parameters = initialize_network_layers(size)
    moments, velocity = initialize_moments(size)

    def train_runner(dataloader, lr_scheduler, t):
        each_batch_loss = []

        for i, (input_image, label) in enumerate(dataloader):
            # Initial activations
            activations = initial_activations(parameters, input_image, label)
            activation_lr = lr_scheduler.step()

            losses = []
            for _ in range(8):
                predicted_activations = forward_pass(activations, parameters)
                # Get the network prediction about the activations and calculate the error between the previous activations
                activations_error = calculate_activation_error(activations, predicted_activations)

                # Inference and Learning Phase
                update_activations(activations, activations_error, parameters, activation_lr)
                update_weights(activations, activations_error, parameters, moments, velocity, parameters_lr, t)

                loss = cross_entropy(label, predicted_activations[-1])
                losses.append(np.mean(loss))

            each_batch_loss.append(sum(losses))

        return np.mean(np.array(each_batch_loss))

    def test_runner(dataloader):
        accuracy = []
        correctness = []
        wrongness = []
        for i, (batched_image, batched_label) in enumerate(dataloader):
            model_output = predict(batched_image, parameters)
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
