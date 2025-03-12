import gzip
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from features import RED, GREEN, RESET
from batching import image_data_batching

class DPC(nn.Module):
    def __init__(self, input_dim, lower_dim=256, higher_dim=64, K=5):
        super().__init__()
        self.K = K
        self.lower_dim = lower_dim
        self.higher_dim = higher_dim

        # Spatial decoder
        self.lower_level_network = nn.Linear(lower_dim, input_dim, bias=False)

        # Transition matrices
        self.Vk = nn.ParameterList([
            nn.Parameter(torch.Tensor(lower_dim, lower_dim)) 
            for _ in range(K)])

        for vk in self.Vk:
            nn.init.kaiming_normal_(vk, nonlinearity='relu')

        # Hypernetwork
        self.hyper_network = nn.Sequential(
            nn.Linear(higher_dim, 128),
            nn.ReLU(),
            nn.Linear(128, K))

        # Higher-level dynamics
        self.higher_rnn = nn.RNNCell(input_dim, higher_dim, nonlinearity='relu')

        # Digit classifier only
        self.digit_classifier = nn.Linear(lower_dim, 10)

    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device

        # Initialize states
        rt = torch.zeros(batch_size, self.lower_dim, device=device)
        rh = torch.zeros(batch_size, self.higher_dim, device=device)

        # Storage for outputs
        pred_errors = []
        digit_logits = []

        for t in range(seq_len):
            # Decode current frame
            pred_xt = self.lower_level_network(rt)
            error = x_seq[:, t] - pred_xt

            # Store prediction error
            pred_errors.append(error.pow(2).mean())

            # Update higher level
            rh = self.higher_rnn(error.detach(), rh)

            # Generate transition weights
            w = self.hyper_network(rh)

            # Combine transition matrices
            V = sum(w[:, k].unsqueeze(-1).unsqueeze(-1) * self.Vk[k] for k in range(self.K))
            
            # Update lower state with ReLU and noise
            rt = F.relu(torch.einsum('bij,bj->bi', V, rt)) + 0.01 * torch.randn_like(rt)

            # Collect digit logits
            digit_logits.append(self.digit_classifier(rt))

        # Average predictions over time
        digit_logit = torch.stack(digit_logits).mean(0)
        pred_error = torch.stack(pred_errors).mean()

        return {'digit_prediction': digit_logit, 'prediction_error': pred_error}
    
def train(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    each_batch_losses = []
    for input_image, digits in loader:
        input_image = input_image.view(input_image.size(0), -1, 28*28).repeat(1, 5, 1)
        digits = digits
        
        outputs = model(input_image)

        # Calculate losses
        loss_digit = loss_fn(outputs['digit_prediction'], digits)
        loss_pred = outputs['prediction_error']
        
        # Combine losses with regularization
        loss = (loss_digit + 0.1 * loss_pred + 0.01 * (model.lower_level_network.weight.pow(2).mean()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        each_batch_losses.append(loss.item())

    return torch.mean(torch.tensor(each_batch_losses)).item()

def evaluate(model, loader):
    model.eval()
    model_accuracies = []
    correctness = []
    wrongness = []

    with torch.no_grad():
        for i, (batched_image, batched_label) in enumerate(loader):
            batched_image = batched_image.view(batched_image.size(0), -1, 28*28).repeat(1, 5, 1)
            model_prediction = model(batched_image)['digit_prediction']
            each_batch_accuracy = (model_prediction.argmax(dim=-1) == batched_label).float().mean().item()
            
            for each in range(len(batched_label)//10):
                model_digit_prediction = model_prediction[each].argmax().item()
                expected_digit = batched_label[each].item()

                if model_digit_prediction == expected_digit:
                    correctness.append((model_digit_prediction, expected_digit))
                else:
                    wrongness.append((model_digit_prediction, expected_digit))

            print(f'Number of test samples: {i+1}\r', end='', flush=True)
            model_accuracies.append(each_batch_accuracy)

        random.shuffle(correctness)
        random.shuffle(wrongness)
        print(f'{GREEN}Model Correct Predictions{RESET}')
        [print(f"Digit Image is: {GREEN}{expected}{RESET} Model Prediction: {GREEN}{prediction}{RESET}") for i, (prediction, expected) in enumerate(correctness) if i < 5]
        print(f'{RED}Model Wrong Predictions{RESET}')
        [print(f"Digit Image is: {RED}{expected}{RESET} Model Prediction: {RED}{prediction}{RESET}") for i, (prediction, expected) in enumerate(wrongness) if i < 5]

    return torch.mean(torch.tensor(model_accuracies)).item()

def relu(input_data, return_derivative=False):
    if return_derivative:
        return np.where(input_data > 0, 1, 0)
    else:
        return np.maximum(0, input_data)

def lower_network_forward(input_data, parameters):
    activation = np.matmul(input_data, parameters.T)

    return activation

def transition_matrices_forward(input_data, parameters):
    pass

def hyper_network_forward(input_data, parameters):
    activations = [input_data]
    activation = input_data
    for each in range(len(parameters)):
        last_layer = each == len(parameters)-1
        weights = parameters[each][0]
        bias = parameters[each][1]

        pre_activation = np.matmul(activation, weights.T) + bias
        activation = pre_activation if last_layer else relu(pre_activation)
        activations.append(activation)
    
    return activations

def higher_rnn_forward(input_data, hidden_state, parameters):
    input_to_hidden_params = parameters[0]
    hidden_to_hidden_params = parameters[1]

    input_to_hidden_activation = np.matmul(input_data, input_to_hidden_params[0].T) + input_to_hidden_params[1]
    hidden_to_hidden_activation = np.matmul(hidden_state, hidden_to_hidden_params[0].T) + hidden_to_hidden_params[1]

    return relu((input_to_hidden_activation + hidden_to_hidden_activation))


def digit_classifier_forward(input_data, parameters):
    pass

def numpy_dpc(torch_model):
    """Shared parameters initialization with torch model"""

    # Spatial decoder
    lower_level_network_parameters = torch_model.lower_level_network.weight.data.numpy()
    # Transition matrices
    Vk_parameters = [vk.data.numpy() for vk in torch_model.Vk]
    # Hypernetwork
    hyper_network_parameters = [[layer.weight.data.numpy(), layer.bias.data.numpy()] for layer in torch_model.hyper_network if isinstance(layer, nn.Linear)]
    # Higher-level dynamics
    higher_rnn_parameters = [[torch_model.higher_rnn.weight_ih.data.numpy(), torch_model.higher_rnn.bias_ih.data.numpy()], [torch_model.higher_rnn.weight_hh.data.numpy(), torch_model.higher_rnn.bias_hh.data.numpy()]]
    # Digit classifier only
    digit_classifier_parameters = [torch_model.digit_classifier.weight.data.numpy(), torch_model.digit_classifier.bias.data.numpy()]

    def forward(batched_image):
        batch_size, seq_len, _ = batched_image.shape

        # Initialize states
        rt = np.zeros(shape=(batch_size, torch_model.lower_dim))
        rh = np.zeros(shape=(batch_size, torch_model.higher_dim))

        # Storage for outputs
        pred_errors = []
        digit_logits = []

        for t in range(seq_len):
            pred_xt = lower_network_forward(rt, lower_level_network_parameters)
            error = batched_image[:, t] - pred_xt

            # Store prediction error
            pred_errors.append(np.mean(error**2))

            # Update higher level
            rh = higher_rnn_forward(error, rh, higher_rnn_parameters)

            # Generate transition weights
            w = hyper_network_forward(rh, hyper_network_parameters)

    def train_runner(dataloader):
        for batched_image, batched_label in dataloader:
            # From (Batch, height*width) to (Batch, 5, height*width)
            batched_image = batched_image.view(batched_image.size(0), -1, 28*28).repeat(1, 5, 1).numpy()
            batched_label = batched_label.numpy()

            forward(batched_image)

    return train_runner


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

    # # Torch Runner
    # for epoch in range(MAX_EPOCHS):
    #     training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
    #     test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
    #     loss = train(torch_model, training_loader)
    #     accuracy = evaluate(torch_model, test_loader)
    #     print('EPOCH: {} LOSS: {} Accuracy: {}'.format(epoch+1, loss, accuracy))

    # Numpy Runner
    for epoch in range(MAX_EPOCHS):
        training_loader = image_data_batching(train_images, train_labels, batch_size=128, shuffle=True)
        test_loader = image_data_batching(test_images, test_labels, batch_size=128, shuffle=True)
        numpy_model(training_loader)

main_runner()
