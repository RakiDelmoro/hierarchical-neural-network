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

    caches['rnn_caches'].append([input_to_hidden_caches, hidden_to_hidden_caches])

    return output

def combine_transitions(weights, Vk_parameters, caches):
    combined_transitions = []
    for k in range(len(Vk_parameters)):
        transition = weights[:, k].reshape(-1, 1, 1) * Vk_parameters[k]
        combined_transitions.append(transition)

    caches['combine_transitions_caches'].append(sum(combined_transitions))

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

def backpropagate(batched_image, caches, gradients, torch_model):
    # Dimensions
    batch_size, seq_len, _ = batched_image.shape
    K = torch_model.K
    lower_dim = torch_model.lower_dim
    higher_dim = torch_model.higher_dim

    # Numpy model activations caches
    lower_network_act = caches['lower_network_caches']
    hyper_network_act = caches['hyper_network_caches']
    rnn_act = caches['rnn_caches']
    combine_transitions_act = caches['combine_transitions_caches']
    classifier_act = caches['classifier_caches']
    prediction_error = caches['prediction_error']

    # Initialize gradients
    lower_level_network_gradients = np.zeros_like(torch_model.lower_level_network.weight.data.numpy())
    Vk_gradients = [np.zeros_like(vk.detach().numpy()) for vk in torch_model.Vk]
    hyper_network_gradients = [[np.zeros_like(layer.weight.data.numpy()), np.zeros_like(layer.bias.data.numpy())] for layer in torch_model.hyper_network if isinstance(layer, nn.Linear)][1]
    higher_rnn_gradients = [[np.zeros_like(torch_model.higher_rnn.weight_ih.data.numpy()), np.zeros_like(torch_model.higher_rnn.bias_ih.data.numpy())],
                             [np.zeros_like(torch_model.higher_rnn.weight_hh.data.numpy()), np.zeros_like(torch_model.higher_rnn.bias_hh.data.numpy())]]
    digit_classifier_gradients = [np.zeros_like(torch_model.digit_classifier.weight.data.numpy()), np.zeros_like(torch_model.digit_classifier.bias.data.numpy())]

def numpy_train_runner(model, dataloader, torch_model):
    for batched_image, batched_label in dataloader:
        # From (Batch, height*width) to (Batch, seq_len, height*width)
        batched_image = batched_image.view(batched_image.size(0), -1, 28*28).repeat(1, 5, 1).numpy()
        batched_label = batched_label.numpy()

        model_outputs, model_activations = model(batched_image)
        digit_prediction = model_outputs['digit_prediction']
        prediction_error = model_outputs['prediction_error']

        # Calculate losses
        loss, gradients = cross_entropy_loss(digit_prediction, batched_label)

        backpropagate(batched_image, model_activations, gradients, torch_model)
        
        # Combine losses with regularization
        loss = loss + 0.1 * prediction_error #+ 0.01 * np.mean(lower_level_network_parameters**2)

    return loss

def numpy_dpc(torch_model):
    """Shared parameters initialization with torch model"""

    forward_caches = {'lower_network_caches': [],'hyper_network_caches': [], 'rnn_caches': [], 'combine_transitions_caches': [], 'classifier_caches': [], 'prediction_error': []}

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

            # Update higher level
            higher_level_state = rnn_forward(error, higher_level_state, higher_rnn_parameters, forward_caches)

            # Generate transition weights
            w = hyper_network_forward(higher_level_state, hyper_network_parameters, forward_caches)
            value = combine_transitions(w, Vk_parameters, forward_caches)

            # Update lower state with ReLU and noise
            lower_level_state_reshaped = np.expand_dims(lower_level_state, axis=-1)
            lower_level_state = relu(np.matmul(value, lower_level_state_reshaped).squeeze(-1)) #+ 0.01 * np.random.randn(*lower_level_state.shape)

            # Collect digit logits
            model_prediction = classifier_forward(lower_level_state, digit_classifier_parameters, forward_caches)

            digit_logits.append(model_prediction)
            # Store frame prediction error
            pred_errors.append(avg_error)

        model_digit_prediction = np.stack(digit_logits).mean(0)
        prediction_error = np.stack(pred_errors).mean()

        return {'digit_prediction': model_digit_prediction, 'prediction_error': prediction_error}, forward_caches

    return forward
