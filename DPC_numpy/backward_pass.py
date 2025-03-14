import numpy as np
import torch.nn as nn
from DPC_numpy.utils import relu

def classifier_backward(seq_len, activation, loss, gradients, parameters):
    # Gradients from digit classifier
    classifier_input, _ = activation
    classifier_gradient = loss / seq_len
    gradients[0] += np.matmul(classifier_gradient.T, classifier_input)
    gradients[1] += np.sum(classifier_gradient, axis=0)

    classifier_grad_propagate = np.matmul(classifier_gradient, parameters[0])

    return classifier_grad_propagate

def lower_net_update_state_backward(state, gradients):
    # Backprop through lower net update state
    lower_net_state, lower_net_reshaped, value = state
    relu_deriv = relu(lower_net_state, return_derivative=True)
    chain_gradient = gradients * relu_deriv
    chain_gradient_reshaped = chain_gradient.reshape(-1, chain_gradient.shape[1], 1)

    # Gradients for value_t and lower_prev
    dL_dvalue = np.einsum('bik,bjk->bij', chain_gradient_reshaped, lower_net_reshaped) / lower_net_state.shape[0]
    dL_d_lower_prev = np.einsum('bij,bik->bjk', value, chain_gradient_reshaped).squeeze()

    return dL_dvalue, dL_d_lower_prev

def combine_transitions_backward(deriv_value, activation, Vk_gradients, parameters, batch_size):
    # Backprop through combine transitions
    generated_weights, _ = activation
    for k in range(len(parameters)):
        Vk_gradients[k] += np.mean(generated_weights[:, k, None, None] * deriv_value, axis=0)

    dL_dgenerated_weights = np.stack([np.sum(vk * deriv_value, axis=(1, 2)) for vk in parameters], axis=1) / batch_size

    return dL_dgenerated_weights

def hyper_network_backward(activation, gradients, hyper_net_gradients, parameters):
    # Backprop through hyper network
    dL_d_hyper = gradients
    for layer_idx in reversed(range(len(parameters))):
        weights, _ = parameters[layer_idx]
        layer_input = activation[layer_idx]
        layer_output = activation[layer_idx + 1]

        if layer_idx == len(parameters) - 1:
            dpre_activation = dL_d_hyper
        else:
            dpre_activation = dL_d_hyper * relu(layer_output, return_derivative=True)

        hyper_net_gradients[layer_idx][0] += np.matmul(dpre_activation.T, layer_input)
        hyper_net_gradients[layer_idx][1] += np.sum(dpre_activation, axis=0)

        dL_d_hyper = np.matmul(dpre_activation, weights)

    return dL_d_hyper

def rnn_backward(activations, previous_gradient, gradients, parameters):
    input_to_hidden_caches, hidden_to_hidden_caches, output = activations
    
    dL_d_activation = previous_gradient * relu(output, return_derivative=True)

    # Input to hidden gradients
    gradients[0][0] += np.matmul(dL_d_activation.T, input_to_hidden_caches[0])
    gradients[0][1] += np.sum(dL_d_activation, axis=0)

    # input to hidden propagate 
    inp_to_hid_propagate_err = np.matmul(dL_d_activation, parameters[0][0])

    # Hidden to hidden gradients
    gradients[1][0] += np.matmul(dL_d_activation.T, hidden_to_hidden_caches[0])
    gradients[1][1] += np.sum(dL_d_activation, axis=0)

    hid_to_hid_propagate_err = np.matmul(dL_d_activation, parameters[1][0])

    return inp_to_hid_propagate_err, hid_to_hid_propagate_err

def backpropagate(batched_image, activations_caches, model_parameters, gradients, pred_frame_gradients, torch_model):
    # Dimensions
    batch_size, seq_len, _ = batched_image.shape
    K = torch_model.K
    lower_dim = torch_model.lower_dim
    higher_dim = torch_model.higher_dim

    # Numpy model activations caches
    lower_network_act = activations_caches['lower_network_caches']
    hyper_network_act = activations_caches['hyper_network_caches']
    rnn_activations = activations_caches['rnn_caches']
    combine_transitions_act = activations_caches['combine_transitions_caches']
    classifier_act = activations_caches['classifier_caches']
    prediction_error = activations_caches['prediction_error']
    lower_net_states = activations_caches['lower_net_states']

    # Numpy model parameters
    lower_net_parameters = model_parameters['lower_network_parameters']
    Vk_parameters = model_parameters['Vk_parameters']
    hyper_network_parameters = model_parameters['hyper_network_parameters']
    rnn_parameters = model_parameters['higher_rnn_parameters']
    digit_classifier_parameters = model_parameters['digit_classifier_parameters']

    # Initialize gradients
    lower_net_gradients = np.zeros_like(torch_model.lower_level_network.weight.data.numpy())
    Vk_gradients = [np.zeros_like(vk.detach().numpy()) for vk in torch_model.Vk]
    hyper_net_gradients = [[np.zeros_like(layer.weight.data.numpy()), np.zeros_like(layer.bias.data.numpy())] for layer in torch_model.hyper_network if isinstance(layer, nn.Linear)]
    rnn_gradients = [[np.zeros_like(torch_model.higher_rnn.weight_ih.data.numpy()), np.zeros_like(torch_model.higher_rnn.bias_ih.data.numpy())],
                             [np.zeros_like(torch_model.higher_rnn.weight_hh.data.numpy()), np.zeros_like(torch_model.higher_rnn.bias_hh.data.numpy())]]
    classifier_gradients = [np.zeros_like(torch_model.digit_classifier.weight.data.numpy()), np.zeros_like(torch_model.digit_classifier.bias.data.numpy())]

    dL_d_higher = 0
    dL_d_lower = 0

    for t in reversed(range(seq_len)):
        # Backprop through digit classifier
        classifier_activation = classifier_act[t]
        classifier_grad_propagate = classifier_backward(seq_len, classifier_activation, gradients, classifier_gradients, digit_classifier_parameters)

        dL_d_lower += classifier_grad_propagate

        # Backprop through lower net state update
        deriv_value, deriv_prev_lower_state = lower_net_update_state_backward(lower_net_states[t], classifier_grad_propagate)

        # Backprop through combine transitions
        dL_dgenerated_weights = combine_transitions_backward(deriv_value, combine_transitions_act[t], Vk_gradients, Vk_parameters, batch_size)

        # Backprop through hyper network
        dL_d_hyper = hyper_network_backward(hyper_network_act[t], dL_dgenerated_weights, hyper_net_gradients, hyper_network_parameters)
        dL_d_higher += dL_d_hyper

        # Backprop through RNN
        input_to_hidden_gradient, hidden_to_hidden_gradient = rnn_backward(rnn_activations[t], dL_d_higher, rnn_gradients, rnn_parameters)
        dL_d_higher = hidden_to_hidden_gradient

        # Gradients from prediction error
        dL_d_predicted = 2 * prediction_error[t] / (batch_size * np.prod(prediction_error[t].shape[1:]) * seq_len) + input_to_hidden_gradient

        # Backprop lower network
        lower_net_input, _ = lower_network_act[t]
        lower_net_gradients += np.matmul(dL_d_predicted.T, lower_net_input)
        dL_d_lower_prev_error = np.matmul(dL_d_predicted, lower_net_parameters)

        # Accumulate lower state gradients
        dL_d_lower = deriv_prev_lower_state + dL_d_lower_prev_error

    gradients = {'lower_net_gradients': lower_net_gradients, 'Vk_gradients': Vk_gradients, 'hyper_net_gradients': hyper_net_gradients, 'rnn_gradients': rnn_gradients, 'classifier_gradient': classifier_gradients}   

    return gradients
