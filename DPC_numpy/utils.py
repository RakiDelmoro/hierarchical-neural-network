import numpy as np

def softmax(input_data):
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
