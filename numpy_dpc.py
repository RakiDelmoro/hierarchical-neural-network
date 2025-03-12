import numpy as np

def dynamic_predictive_coding(input_dimension, lower_dim=256, higher_dim=64, K=5):
    # Spatial decoder
    lower_level_network = np.random.randn(lower_dim, input_dimension)

    # Transition matrices
    Vk = [np.random.randn(lower_dim, lower_dim) for _ in range(K)]

    # Hypernetwork
    hyper_network = [
        np.random.randn(128, higher_dim),
        np.random.randn(K, 128)
    ]

    # Higher-level dynamics
    higher_rnn = np.random.randn(input_dimension, higher_dim)

    # Digit classifier only
    digit_classifier = np.random.randn(lower_dim, 10)

    return lower_level_network, Vk, hyper_network, higher_rnn, digit_classifier