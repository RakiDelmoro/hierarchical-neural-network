import gzip
import pickle
import numpy as np
from dataloader.utilities import encode, decode

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
