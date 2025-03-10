import torch
import torch.nn as nn
import torch.nn.functional as F

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
            for _ in range(K)
        ])
        for vk in self.Vk:
            nn.init.kaiming_normal_(vk, nonlinearity='relu')
        
        # Hypernetwork
        self.hyper_network = nn.Sequential(
            nn.Linear(higher_dim, 128),
            nn.ReLU(),
            nn.Linear(128, K)
        )
        
        # Higher-level dynamics
        self.higher_rnn = nn.GRUCell(input_dim, higher_dim)
        
        # 0-9 Classifier (For predicting digit)
        self.digit_classifier = nn.Linear(lower_dim, 10)
        # left-right-up-down classifier (For predicting next frame)
        self.direction_classifier = nn.Linear(higher_dim, 4)
        
    def forward(self, x_seq):
        batch_size, seq_len, input_dim = x_seq.shape
        device = x_seq.device
        
        # Initialize states
        rt = torch.zeros(batch_size, self.lower_dim, device=device)
        rh = torch.zeros(batch_size, self.higher_dim, device=device)

        # Storage for outputs
        pred_errors = []
        digit_logits = []
        direction_logits = []

        for t in range(seq_len):
            # Decode current frame
            pred_xt = self.lower_level_network(rt)
            error = x_seq[:, t] - pred_xt
    
            # Store prediction error (Mean Squared Error)
            pred_errors.append(error.pow(2).mean())
    
            # Update higher level
            rh = self.higher_rnn(error.detach(), rh)
            
            # Generate transition weights
            w = self.hyper_network(rh)
            
            # Combine transition matrices
            V = sum(w[:, k].unsqueeze(-1).unsqueeze(-1) * self.Vk[k] 
                     for k in range(self.K))
            
            # Update lower state with ReLU and noise
            rt = F.relu(torch.einsum('bij,bj->bi', V, rt)) + 0.01 * torch.randn_like(rt)
            
            # Classify (using current states)
            digit_logits.append(self.digit_classifier(rt))
            direction_logits.append(self.direction_classifier(rh))
        
        # Average predictions over time
        digit_logit = torch.stack(digit_logits).mean(0)
        direction_logit = torch.stack(direction_logits).mean(0)
        pred_error = torch.stack(pred_errors).mean()
        
        return {
            'digit_prediction': digit_logit,
            'direction_frame': direction_logit,
            'prediction_error': pred_error
        }
    
input_x = torch.randn(32, 10, 784)
model = DPC(784)
model(input_x)