import librosa
import numpy as np

str_to_int = lambda chars: {ch:i for i,ch in enumerate(chars)}
int_to_str = lambda chars: {i:ch for i,ch in enumerate(chars)}

def encode(text, unique_chars: list):
    str_to_int_map = {ch:i for i,ch in enumerate(unique_chars)}
    return [str_to_int_map[c] for c in text]

def decode(indices, unique_chars: list):
    int_to_str_map = {i:ch for i,ch in enumerate(unique_chars)}
    return [int_to_str_map[i] for i in indices]

def text_label_one_hot(label_arr):
    one_hot_expected = np.zeros(shape=(label_arr.shape[0], 65))
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
        yield img_arr[train_indices[start_idx:end_idx]], text_label_one_hot(label_arr[train_indices[start_idx:end_idx]])

def audio_data_batching(audio_arr, label_arr, batch_size, shuffle):
    num_samples = audio_arr.shape[0]
    samples_indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(samples_indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        yield audio_arr[samples_indices[start_idx:end_idx]], label_arr[samples_indices[start_idx:end_idx]]


def load_audio_file(path, max_pad_len=180):
    audio, sample_rate = librosa.load(path, sr=22050)
    n_fft = min(2048, len(audio))
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft, hop_length=n_fft//4)

    # Pad if necessary
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # If the MFCC features are longer than max_pad_len, truncate them
        mfccs = mfccs[:, :max_pad_len]

    return mfccs.T

def audio_label_one_hot():
    pass

def pad_audio_sequence(array_seq, max_length=100, dtype=np.float32):
    pad_width = [(0, max_length - array_seq.shape[0])]
    pad_width.extend((0,0) for _ in range(array_seq.ndim - 1))
    return np.pad(array_seq, pad_width, mode='constant')

def norm_audio_array(arr):
    max_abs = np.max(np.abs(arr))
    if max_abs == 0: return arr
    return arr / max_abs
