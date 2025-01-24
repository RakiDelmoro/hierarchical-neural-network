from dataloader.utilities import image_data_batching, text_data_batching
from dataloader.fetch_data import image_classification_task, text_generation_task

def runner():
    # Mnist dataset
    train_image, train_labels, test_image, test_labels =  image_classification_task('./datasets/mnist.pkl.gz')
    # Text dataset
    train_text, test_text, vocab_size = text_generation_task('./datasets/nlp-task.txt')

    for _ in range(10):
        # Image classification task dataloader
        image_training_set = image_data_batching(train_image, train_labels, batch_size=32, shuffle=True)
        image_testing_set = image_data_batching(test_image, test_labels, batch_size=32, shuffle=True)

        # Text generation task dataloader
        text_training_set = text_data_batching(train_text, batch_size=32, max_context_length=8)
        text_testing_set = text_data_batching(test_text, batch_size=32, max_context_length=8)

runner()
