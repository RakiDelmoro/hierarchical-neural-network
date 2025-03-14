import numpy as np
import torch.nn as nn

def softmax(input_data, return_derivative=False):
    # Subtract max value for numerical stability
    shifted_data = input_data - np.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = np.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    return exp_data / sum_exp_data

def cross_entropy_loss(predicted, expected):
    predicted_probs = softmax(predicted)
    one_hot_expected = np.zeros(shape=(predicted.shape[0], 10))
    one_hot_expected[np.arange(len(expected)), expected] = 1

    loss_gradients = (predicted_probs - one_hot_expected)

    return -np.mean(np.sum(one_hot_expected * np.log(predicted_probs), axis=-1)), loss_gradients

def relu(input_data, return_derivative=False):
    if return_derivative:
        return np.where(input_data > 0, 1, 0)
    else:
        return np.maximum(0, input_data)

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

def classifier_forward(input_data, parameters, caches):
    activations = [input_data]
    weights = parameters[0]
    bias = parameters[1]
    
    activation = np.matmul(input_data, weights.T) + bias
    activations.append(activation)

    caches['classifier_caches'].append(activations)

    return activation

def prediction_frame_error(predicted, expected, caches):
    error = predicted - expected
    caches['prediction_error'].append(error)

    return np.mean(error**2), error

def lower_net_state_update(updated_lower_net_state, value, caches):
    lower_net_reshaped = np.expand_dims(updated_lower_net_state, axis=-1)
    updated_lower_net_state = relu(np.matmul(value, lower_net_reshaped).squeeze(-1)) #+ 0.01 * np.random.randn(*updated_lower_net_state.shape)

    caches['lower_net_states'].append([updated_lower_net_state, lower_net_reshaped, value])

    return updated_lower_net_state

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
        dL_d_predicted = 2 * prediction_error[t] / (batch_size * np.prod(prediction_error[t].shape[1:]) / seq_len) + input_to_hidden_gradient

        # Backprop lower network
        lower_net_input, _ = lower_network_act[t]
        lower_net_gradients += np.matmul(dL_d_predicted.T, lower_net_input)
        dL_d_lower_prev_error = np.matmul(dL_d_predicted, lower_net_parameters)

        # Accumulate lower state gradients
        dL_d_lower = deriv_prev_lower_state + dL_d_lower_prev_error

    gradients = {'lower_net_gradients': lower_net_gradients, 'Vk_gradients': Vk_gradients, 'hyper_net_gradients': hyper_net_gradients, 'rnn_gradients': rnn_gradients, 'classifier_gradient': classifier_gradients}   

    return gradients 

def update_parameters(gradients, parameters, lr, batch_size):
    # Gradients
    lower_net_gradients = gradients['lower_net_gradients']
    Vk_gradients = gradients['Vk_gradients']
    hyper_net_gradients = gradients['hyper_net_gradients']
    rnn_gradients = gradients['rnn_gradients']
    classifier_gradients = gradients['classifier_gradient']

    # Parameters
    lower_net_parameters = parameters['lower_network_parameters']
    Vk_parameters = parameters['Vk_parameters']
    hyper_network_parameters = parameters['hyper_network_parameters']
    rnn_parameters = parameters['higher_rnn_parameters']
    digit_classifier_parameters = parameters['digit_classifier_parameters']

    digit_classifier_parameters[0] -= lr * (classifier_gradients[0] / batch_size)
    digit_classifier_parameters[1] -= lr * (classifier_gradients[1] / batch_size)

    rnn_parameters[0][0] -= lr * (rnn_gradients[0][0] / batch_size)
    rnn_parameters[0][1] -= lr * (rnn_gradients[0][1] / batch_size)
    rnn_parameters[1][0] -= lr * (rnn_gradients[1][0] / batch_size)
    rnn_parameters[1][1] -= lr * (rnn_gradients[1][1] / batch_size)

    hyper_network_parameters[0][0] -= lr * (hyper_net_gradients[0][0] / batch_size)
    hyper_network_parameters[0][1] -= lr * (hyper_net_gradients[0][1] / batch_size)
    hyper_network_parameters[1][0] -= lr * (hyper_net_gradients[1][0] / batch_size)
    hyper_network_parameters[1][1] -= lr * (hyper_net_gradients[1][1] / batch_size)

    for each in range(len(Vk_parameters)): Vk_parameters[each] -= lr * (Vk_gradients[each] / batch_size)

    lower_net_parameters -= lr * (lower_net_gradients / batch_size)

def numpy_train_runner(model, dataloader, torch_model):
    for batched_image, batched_label in dataloader:
        # From (Batch, height*width) to (Batch, seq_len, height*width)
        batched_image = batched_image.view(batched_image.size(0), -1, 28*28).repeat(1, 5, 1).numpy()
        batched_label = batched_label.numpy()

        model_outputs, model_activations, model_parameters = model(batched_image)
        digit_prediction = model_outputs['digit_prediction']
        prediction_error = model_outputs['prediction_error']

        # Calculate losses
        loss, digit_pred_gradients = cross_entropy_loss(digit_prediction, batched_label)

        gradients = backpropagate(batched_image, model_activations, model_parameters, digit_pred_gradients, prediction_error, torch_model)
        update_parameters(gradients, model_parameters, 0.01, batched_image.shape[0])

        # Combine losses with regularization
        loss = loss + 0.1 * prediction_error #+ 0.01 * np.mean(lower_level_network_parameters**2)

    return loss

def numpy_dpc(torch_model):
    """Shared parameters initialization with torch model"""

    forward_caches = {'lower_network_caches': [],'hyper_network_caches': [], 'rnn_caches': [], 'combine_transitions_caches': [], 'lower_net_states': [], 'classifier_caches': [], 'prediction_error': []}

    # Spatial decoder
    lower_level_network_parameters = torch_model.lower_level_network.weight.data.numpy()
    # Transition matrices
    Vk_parameters = [vk.data.numpy() for vk in torch_model.Vk]
    # Hypernetwork
    hyper_network_parameters = [[layer.weight.data.numpy(), layer.bias.data.numpy()] for layer in torch_model.hyper_network if isinstance(layer, nn.Linear)]
    # Higher-level dynamics
    higher_rnn_parameters = [[torch_model.higher_rnn.weight_ih.data.numpy(), torch_model.higher_rnn.bias_ih.data.numpy()], [torch_model.higher_rnn.weight_hh.data.numpy(), torch_model.higher_rnn.bias_hh.data.numpy()]]
    # Digit classifier only
    digit_classifier_parameters = [torch_model.digit_classifier.weight.data.numpy(), torch_model.digit_classifier.bias.data.numpy()]

    parameters_caches = {'lower_network_parameters': lower_level_network_parameters, 'Vk_parameters': Vk_parameters, 'hyper_network_parameters': hyper_network_parameters, 'higher_rnn_parameters': higher_rnn_parameters, 'digit_classifier_parameters': digit_classifier_parameters}

    def forward(batched_image):
        batch_size, seq_len, _ = batched_image.shape

        # Initialize states
        lower_level_state = np.zeros(shape=(batch_size, torch_model.lower_dim))
        higher_level_state = np.zeros(shape=(batch_size, torch_model.higher_dim))

        # Storage for outputs
        pred_errors = []
        digit_logits = []

        for t in range(seq_len):
            each_frame = batched_image[:, t]

            predicted_frame = lower_network_forward(lower_level_state, lower_level_network_parameters, forward_caches)
            avg_error, error = prediction_frame_error(predicted_frame, each_frame, forward_caches)

            # Use RnnCell to update the higher level state
            higher_level_state = rnn_forward(error, higher_level_state, higher_rnn_parameters, forward_caches)

            # Generate transition weights
            generated_weights = hyper_network_forward(higher_level_state, hyper_network_parameters, forward_caches)
            value = combine_transitions(generated_weights, Vk_parameters, forward_caches)

            # Update lower state with ReLU and noise
            lower_level_state = lower_net_state_update(lower_level_state, value, forward_caches)

            # Collect digit logits
            model_prediction = classifier_forward(lower_level_state, digit_classifier_parameters, forward_caches)

            digit_logits.append(model_prediction)
            # Store frame prediction error
            pred_errors.append(avg_error)

        model_digit_prediction = np.stack(digit_logits).mean(0)
        prediction_error = np.stack(pred_errors).mean()

        return {'digit_prediction': model_digit_prediction, 'prediction_error': prediction_error}, forward_caches, parameters_caches

    return forward
