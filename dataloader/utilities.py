import numpy as np

str_to_int = lambda chars: {ch:i for i,ch in enumerate(chars)}
int_to_str = lambda chars: {i:ch for i,ch in enumerate(chars)}

def encode(text, unique_chars: list):
    str_to_int_map = {ch:i for i,ch in enumerate(unique_chars)}
    return [str_to_int_map[c] for c in text]

def decode(indices, unique_chars: list):
    int_to_str_map = {i:ch for i,ch in enumerate(unique_chars)}
    return [int_to_str_map[i] for i in indices]

def one_hot_encoded(label_arr):
    one_hot_expected = np.zeros(shape=(label_arr.shape[0], 10))
    one_hot_expected[np.arange(len(label_arr)), label_arr] = 1
    return one_hot_expected

def text_data_batching(indices_arr, batch_size, max_context_length):
    ix = np.random.randint(low=len(indices_arr) - max_context_length, size=batch_size)
    input_indices = np.stack([indices_arr[i:i+max_context_length] for i in ix])
    next_indices = np.stack([indices_arr[i+1:i+max_context_length+1] for i in ix])
    return input_indices, next_indices

def image_data_batching(img_arr, label_arr, batch_size, shuffle):
    num_train_samples = img_arr.shape[0]    
    # Total samples
    train_indices = np.arange(num_train_samples)
    if shuffle: np.random.shuffle(train_indices)

    for start_idx in range(0, num_train_samples, batch_size):
        end_idx = start_idx + batch_size
        yield img_arr[train_indices[start_idx:end_idx]], one_hot_encoded(label_arr[train_indices[start_idx:end_idx]])
