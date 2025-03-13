import gzip
import pickle
from sandbox import DPC, train, evaluate
from dpc_numpy import numpy_dpc
from dpc_numpy import numpy_train_runner
from batching import image_data_batching

def main_runner():
    MAX_EPOCHS = 100
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    with gzip.open('./datasets/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    torch_model = DPC(IMAGE_HEIGHT*IMAGE_WIDTH)
    numpy_model = numpy_dpc(torch_model)

    # Torch Runner
    for epoch in range(MAX_EPOCHS):
        training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
        loss = train(torch_model, numpy_model, training_loader)
        accuracy = evaluate(torch_model, test_loader)
        print('EPOCH: {} LOSS: {} Accuracy: {}'.format(epoch+1, loss, accuracy))

    # Numpy Runner
    for epoch in range(MAX_EPOCHS):
        training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
        loss = numpy_train_runner(numpy_model, training_loader)

main_runner()