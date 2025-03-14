import numpy as np
from DPC_numpy.utils import relu

def lower_network_forward(input_data, parameters, caches):
    activations = [input_data]

    activation = np.matmul(input_data, parameters.T)
    activations.append(activation)

    caches['lower_network_caches'].append(activations)

    return activation

def hyper_network_forward(input_data, parameters, caches):
    activations = [input_data]
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        bias = parameters[each][1]

        pre_activation = np.matmul(activation, weights.T) + bias
        activation = pre_activation if last_layer else relu(pre_activation)
        activations.append(activation)
    
    caches['hyper_network_caches'].append(activations)

    return activation

def rnn_forward(input_data, hidden_state, parameters, caches):
    input_to_hidden_caches = [input_data]
    hidden_to_hidden_caches = [hidden_state]

    input_to_hidden_params = parameters[0]
    hidden_to_hidden_params = parameters[1]

    input_to_hidden_activation = np.matmul(input_data, input_to_hidden_params[0].T) + input_to_hidden_params[1]
    hidden_to_hidden_activation = np.matmul(hidden_state, hidden_to_hidden_params[0].T) + hidden_to_hidden_params[1]

    output = relu((input_to_hidden_activation + hidden_to_hidden_activation))

    input_to_hidden_caches.append(input_to_hidden_activation)
    hidden_to_hidden_caches.append(hidden_to_hidden_activation)

    caches['rnn_caches'].append([input_to_hidden_caches, hidden_to_hidden_caches, output])

    return output

def combine_transitions(weights, Vk_parameters, caches):
    combined_transitions = []
    for k in range(len(Vk_parameters)):
        transition = weights[:, k].reshape(-1, 1, 1) * Vk_parameters[k]
        combined_transitions.append(transition)

    caches['combine_transitions_caches'].append([weights, sum(combined_transitions)])

    return sum(combined_transitions)

def prediction_frame_error(predicted, expected, caches):
    error = predicted - expected
    caches['prediction_error'].append(error)

    return np.mean(error**2), error

def lower_net_state_update(updated_lower_net_state, value, caches):
    lower_net_reshaped = np.expand_dims(updated_lower_net_state, axis=-1)
    activation = relu(np.matmul(value, lower_net_reshaped).squeeze(-1))
    updated_lower_net_state = activation + 0.01 * np.random.randn(*updated_lower_net_state.shape)

    caches['lower_net_states'].append([activation, lower_net_reshaped, value])

    return updated_lower_net_state

def classifier_forward(input_data, parameters, caches):
    activations = [input_data]
    weights = parameters[0]
    bias = parameters[1]
    
    activation = np.matmul(input_data, weights.T) + bias
    activations.append(activation)

    caches['classifier_caches'].append(activations)

    return activation
