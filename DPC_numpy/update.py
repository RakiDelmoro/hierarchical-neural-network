import numpy as np

def update_parameters(gradients, parameters, lr, batch_size):
    # Adam hyperparameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-3

    # Update each parameter group with Adam
    # Initialize timestep
    t = parameters.get('t', 1)

    # Digit Classifier Parameters
    digit_classifier_parameters = parameters['digit_classifier_parameters']
    classifier_gradients = gradients['classifier_gradient']
    if 'digit_classifier_m' not in parameters:
        parameters['digit_classifier_m'] = [np.zeros_like(p) for p in digit_classifier_parameters]
        parameters['digit_classifier_v'] = [np.zeros_like(p) for p in digit_classifier_parameters]
    m = parameters['digit_classifier_m']
    v = parameters['digit_classifier_v']
    for i in range(len(digit_classifier_parameters)):
        grad = classifier_gradients[i] / batch_size
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * (grad**2)
        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)
        digit_classifier_parameters[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    parameters['digit_classifier_m'] = m
    parameters['digit_classifier_v'] = v

    # Higher RNN Parameters
    rnn_parameters = parameters['higher_rnn_parameters']
    rnn_gradients = gradients['rnn_gradients']
    if 'rnn_m' not in parameters:
        parameters['rnn_m'] = [[np.zeros_like(p) for p in layer] for layer in rnn_parameters]
        parameters['rnn_v'] = [[np.zeros_like(p) for p in layer] for layer in rnn_parameters]
    m = parameters['rnn_m']
    v = parameters['rnn_v']
    for layer_idx in range(len(rnn_parameters)):
        for param_idx in range(len(rnn_parameters[layer_idx])):
            grad = rnn_gradients[layer_idx][param_idx] / batch_size
            m[layer_idx][param_idx] = beta1 * m[layer_idx][param_idx] + (1 - beta1) * grad
            v[layer_idx][param_idx] = beta2 * v[layer_idx][param_idx] + (1 - beta2) * (grad**2)
            m_hat = m[layer_idx][param_idx] / (1 - beta1**t)
            v_hat = v[layer_idx][param_idx] / (1 - beta2**t)
            rnn_parameters[layer_idx][param_idx] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    parameters['rnn_m'] = m
    parameters['rnn_v'] = v

    # Hyper Network Parameters
    hyper_network_parameters = parameters['hyper_network_parameters']
    hyper_net_gradients = gradients['hyper_net_gradients']
    if 'hyper_m' not in parameters:
        parameters['hyper_m'] = [[np.zeros_like(p) for p in layer] for layer in hyper_network_parameters]
        parameters['hyper_v'] = [[np.zeros_like(p) for p in layer] for layer in hyper_network_parameters]
    m = parameters['hyper_m']
    v = parameters['hyper_v']
    for layer_idx in range(len(hyper_network_parameters)):
        for param_idx in range(len(hyper_network_parameters[layer_idx])):
            grad = hyper_net_gradients[layer_idx][param_idx] / batch_size
            m[layer_idx][param_idx] = beta1 * m[layer_idx][param_idx] + (1 - beta1) * grad
            v[layer_idx][param_idx] = beta2 * v[layer_idx][param_idx] + (1 - beta2) * (grad**2)
            m_hat = m[layer_idx][param_idx] / (1 - beta1**t)
            v_hat = v[layer_idx][param_idx] / (1 - beta2**t)
            hyper_network_parameters[layer_idx][param_idx] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    parameters['hyper_m'] = m
    parameters['hyper_v'] = v

    # Vk Parameters
    Vk_parameters = parameters['Vk_parameters']
    Vk_gradients = gradients['Vk_gradients']
    if 'Vk_m' not in parameters:
        parameters['Vk_m'] = [np.zeros_like(p) for p in Vk_parameters]
        parameters['Vk_v'] = [np.zeros_like(p) for p in Vk_parameters]
    m = parameters['Vk_m']
    v = parameters['Vk_v']
    for i in range(len(Vk_parameters)):
        grad = Vk_gradients[i] / batch_size
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * (grad**2)
        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)
        Vk_parameters[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    parameters['Vk_m'] = m
    parameters['Vk_v'] = v

    # Lower Network Parameters
    lower_net_parameters = parameters['lower_network_parameters']
    lower_net_gradients = gradients['lower_net_gradients']
    if 'lower_m' not in parameters:
        parameters['lower_m'] = np.zeros_like(lower_net_parameters)
        parameters['lower_v'] = np.zeros_like(lower_net_parameters)
    m = parameters['lower_m']
    v = parameters['lower_v']
    grad = lower_net_gradients / batch_size
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    lower_net_parameters -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    parameters['lower_m'] = m
    parameters['lower_v'] = v

    # Update timestep
    parameters['t'] = t + 1
