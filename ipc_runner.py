import gzip
import pickle
from IPC.utils import image_data_batching
from IPC.model_v2 import ipc_neural_network
from IPC.model_v3 import ipc_neural_network_v3
from IPC.model_v4 import ipc_neural_network_v4
from IPC.model_v5 import ipc_neural_network_v5
from Backprop.model import backprop_neural_network

def runner():
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    PARAMS_LR = 0.0001
    ACTIVATION_LR = 0.1
    NUM_ITERATIONS = 8

    with gzip.open('./datasets/mnist.pkl.gz', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH
    
    ipc_train_runner, ipc_test_runner = ipc_neural_network([784, 256, 256, 10])
    backprop_train_runner, backprop_test_runner = backprop_neural_network([784, 64, 64, 10])
    ipc_v3_train, ipc_v3_test = ipc_neural_network_v3([784, 64, 64, 64, 10], PARAMS_LR, ACTIVATION_LR, NUM_ITERATIONS)
    ipc_v4_train, ipc_v4_test = ipc_neural_network_v4([10, 64, 64, 784])
    ipc_v5_train, ipc_v5_test = ipc_neural_network_v5([784, 64, 64, 10], PARAMS_LR)

    # t = 0
    # IPC V3 Runner
    # for i in range(3000):
    #     training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
    #     test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
    #     loss = ipc_v3_train(training_loader, lr_scheduler, t)
    #     accuracy = ipc_v3_test(test_loader)
    #     print(f'EPOCH: {i+1} LOSS: {loss} Accuracy: {accuracy}')
    #     t += 1

    # t = 0
    # # # IPC V3 Runner
    # for i in range(3000):
    #     training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
    #     test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
    #     loss = ipc_v5_train(training_loader, lr_scheduler, t)
    #     accuracy = ipc_v5_test(test_loader)
    #     print(f'EPOCH: {i+1} LOSS: {loss} Accuracy: {accuracy}')
    #     t += 1

    # for i in range(3000):
    #     training_loader = image_data_batching(train_images, train_labels, batch_size=2098, shuffle=True)
    #     test_loader = image_data_batching(test_images, test_labels, batch_size=2098, shuffle=True)
    #     loss = ipc_v4_train(training_loader)
    #     accuracy = ipc_v4_test(test_loader)
    #     print(f'EPOCH: {i+1} LOSS: {loss} Accuracy: {accuracy}')

    # IPC Runner
    # for i in range(3000):
    #     training_loader = image_data_batching(train_images, train_labels, batch_size=2098, shuffle=True)
    #     test_loader = image_data_batching(test_images, test_labels, batch_size=2098, shuffle=True)
    #     loss = ipc_train_runner(training_loader)
    #     accuracy = ipc_test_runner(test_loader)
    #     print(f'EPOCH: {i+1} Loss: {loss} Accuracy: {accuracy}')

    # Backprop Runner 
    t = 0
    for i in range(3000):
        training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
        loss = backprop_train_runner(training_loader, t)
        accuracy = backprop_test_runner(test_loader)
        print(f'EPOCH: {i+1} Loss: {loss} Accuracy: {accuracy}')
        t += 1

runner()
