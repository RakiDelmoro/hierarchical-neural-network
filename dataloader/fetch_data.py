import gzip
import pickle
import numpy as np
import pandas as pd
from dataloader.utilities import encode, decode, load_audio_file, norm_audio_array, pad_audio_sequence

def text_generation_task(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    # unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    text_data = np.array(encode(text, chars), dtype=np.longlong)
    samples_split = int(0.9*len(text_data))
    train_data = text_data[:samples_split]
    test_data = text_data[samples_split:]
    return train_data, test_data, vocab_size

def image_classification_task(file_path: str):
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    with gzip.open(file_path, 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH
    return train_images, train_labels, test_images, test_labels

def audio_classification_task(file_path: str):
    features = []
    expected_labels = []
    csv_path = file_path + '/audio_table.csv'
    table_data = pd.read_csv(csv_path)
    total_idx_label = len(set(table_data['classID']))
    for i in range(len(table_data)):
        audio_path = file_path + '/audio/fold' + str(table_data['fold'][i]) + '/' + str(table_data['slice_file_name'][i])
        expected_idx = table_data['classID'][i]
        array = load_audio_file(audio_path)
        norm_array = norm_audio_array(array)

        one_hot = np.zeros(total_idx_label)
        one_hot[expected_idx] = 1

        features.append(norm_array)
        expected_labels.append(one_hot)

    audio_arrays = np.array(features)
    expected_arrays = np.array(expected_labels)
    samples_split = int(0.9*len(audio_arrays))
    features_train, label_train = audio_arrays[:samples_split], expected_arrays[:samples_split]
    features_test, label_test = audio_arrays[samples_split:], expected_arrays[samples_split:]

    return features_train, label_train, features_test, label_test
