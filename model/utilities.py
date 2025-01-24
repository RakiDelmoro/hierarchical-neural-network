import math
import torch
import numpy as np
from torch.nn import init
from neurons import linear_neurons
from torch.nn.init import kaiming_uniform_

def dataloader(image_arrays, label_arrays, batch_size: int, shuffle: bool):
    num_samples = image_arrays.shape[0]
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield image_arrays[indices[start:end]], label_arrays[indices[start:end]]

def one_hot_encoded(y_train):
    one_hot_expected = np.zeros(shape=(y_train.shape[0], 10))
    one_hot_expected[np.arange(len(y_train)), y_train] = 1
    return one_hot_expected

def init_params(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = init.uniform_(gen_b_matrix, -bound, bound)
    return [np.array(weights), np.array(bias)]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def neuron(parameters):

    def forward_pass(input_neurons):
        activation = input_neurons
        memories = [input_neurons]
        for each in range(len(parameters)):
            each_params = parameters[each]
            activation = linear_neurons(activation, each_params)
            memories.append(activation)

        return activation, memories

    return forward_pass

def init_model_parameters(neuron_properties):
    neuron_parameters = []
    for size in range(len(neuron_properties)-1):
        input_size = neuron_properties[size]
        output_size = neuron_properties[size+1]
        parameters = init_params(input_size, output_size)
        neuron_parameters.append(parameters)

    return neuron_parameters

def init_weights_stress_transport(neuron_properties):
    weights_stress_transport = []
    for size in range(len(neuron_properties)-1):
        weights = np.random.rand(1, neuron_properties[-(size+1)])
        weights_stress_transport.append(weights)

    return weights_stress_transport