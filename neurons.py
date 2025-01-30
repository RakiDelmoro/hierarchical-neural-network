import numpy as np

def recurrent_neurons(input_neurons, inp_to_memory_params, memory_to_memory_params, starting_memory=None, batch_first=True):
    # Shape correction
    if batch_first: input_neurons = input_neurons.transpose(1, 0, 2)
    seq_len, batch, _ = input_neurons.shape

    if starting_memory is None: starting_memory = np.zeros(shape=(batch, memory_to_memory_params[0].shape[0]))

    # parameters
    weight_ih, bias_ih = inp_to_memory_params
    weight_hh, bias_hh = memory_to_memory_params

    previous_memory = starting_memory
    memories = [starting_memory]
    for t in range(seq_len):
        current_memory_state = np.matmul(input_neurons[t], weight_ih) + bias_ih
        previous_memory_state = np.matmul(previous_memory, weight_hh) + bias_hh
        activation = np.tanh(current_memory_state + previous_memory_state)
        memories.append(activation)
        previous_memory = activation
    rnn_output = np.stack(memories[1:])
    if batch_first: rnn_output = rnn_output.transpose(1, 0, 2)
    return rnn_output, memories

def linear_neurons(input_neurons, parameters):
    weights, bias = parameters
    return np.matmul(input_neurons, weights) + bias
