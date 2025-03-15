import numpy as np
import torch.nn as nn
from DPC_numpy.update import update_parameters
from DPC_numpy.utils import cross_entropy_loss
from DPC_numpy.backward_pass import backpropagate
from DPC_numpy.forward_pass import lower_network_forward, rnn_forward, hyper_network_forward, lower_net_state_update, classifier_forward, prediction_frame_error, combine_transitions

def numpy_train_runner(model, dataloader, torch_model, t):
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
        update_parameters(gradients, model_parameters, 1e-3, batched_image.shape[0], t)

        loss = loss + 0.1 * prediction_error 
        print(loss)
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
        lower_level_state = np.zeros(shape=(batch_size, torch_model.lower_dim), dtype=np.float32)
        higher_level_state = np.zeros(shape=(batch_size, torch_model.higher_dim), dtype=np.float32)

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
