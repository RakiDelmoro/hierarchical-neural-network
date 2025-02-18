import gzip
import pickle
from IPC.model import neural_network
from IPC.utils import image_data_batching


def runner():
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    with gzip.open('./datasets/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH
    
    train_runner, test_runner = neural_network([784, 600, 600, 10])

    for i in range(100):
        training_loader = image_data_batching(train_images, train_labels, batch_size=2098, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=2098, shuffle=True)
        loss = train_runner(training_loader)
        accuracy = test_runner(test_loader)
        print(f'EPOCH: {i+1} Loss: {loss} Accuracy: {accuracy}')

runner()
