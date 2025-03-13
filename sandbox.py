import gzip
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from features import RED, GREEN, RESET
from dpc_numpy import cross_entropy_loss

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
            rt = F.relu(torch.einsum('bij,bj->bi', V, rt)) #+ 0.01 * torch.randn_like(rt)

            # Collect digit logits
            digit_logits.append(self.digit_classifier(rt))

        # Average predictions over time
        digit_logit = torch.stack(digit_logits).mean(0)
        pred_error = torch.stack(pred_errors).mean()

        return {'digit_prediction': digit_logit, 'prediction_error': pred_error}

def train(torch_model, numpy_model, loader):
    optimizer = torch.optim.AdamW(torch_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    each_batch_losses = []
    for input_image, digits in loader:
        input_image = input_image.view(input_image.size(0), -1, 28*28).repeat(1, 5, 1)
        digits = digits
        
        torch_outputs = torch_model(input_image)
        numpy_outputs = numpy_model(input_image.numpy()) # For Debugging (Comparing the result of torch and numpy)

        # Calculate losses
        loss_digit = loss_fn(torch_outputs['digit_prediction'], digits)
        loss_pred = torch_outputs['prediction_error']

        numpy_loss = cross_entropy_loss(numpy_outputs['digit_prediction'], digits.numpy()) # For Debugging (Comparing the result of torch and numpy)

        # Combine losses with regularization
        loss = (loss_digit + 0.1 * loss_pred + 0.01 * (torch_model.lower_level_network.weight.pow(2).mean()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
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



