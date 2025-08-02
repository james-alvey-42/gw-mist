import torch
from models.online_norm import OnlineStandardizingLayer
from models.resnet_1d import ResidualNet

class Network_epsilon(torch.nn.Module):
    def __init__(self, Nbins):
        super().__init__()
        
        # Epsilon network part
        self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        # New MLP to predict the 3 parameters of the Gaussian mu signal
        self.mu_predictor = torch.nn.Sequential(
            torch.nn.Linear(Nbins, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3) # 3 params: amplitude, mean, std
        )
        
        # Register bins as a buffer to ensure it moves to the correct device
        self.register_buffer('bins', torch.arange(Nbins).float())
                
    def epsilon(self, x):
        x = self.net(x.unsqueeze(1)).squeeze(1) # x-net
        return x
    
    def snr(self, x):
        return self.epsilon(x) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        # Adaptive data generation
        ni = x['ni']
        
        ###########################################
        epsilon_sim =  (self.bounds() * torch.rand_like(x['x0'],
                                                    device= x['x0'].device, 
                                                    dtype= x['x0'].dtype)) * ni
        ###########################################
        
        # This is the full data signal, with the true mu (x0) and simulated noise
        data =  x['x0'] + epsilon_sim * ni
        
        # 1. Predict mu parameters from the data
        mu_params = self.mu_predictor(data)
        
        # Unpack parameters: [amplitude, mean, log_std]
        amp = mu_params[:, 0].unsqueeze(1)
        mean = mu_params[:, 1].unsqueeze(1)
        # Use exp to ensure standard deviation is always positive
        std = torch.exp(mu_params[:, 2]).unsqueeze(1)

        # 2. Reconstruct the predicted mu signal (Gaussian)
        bins = self.bins.unsqueeze(0) # Shape: [1, Nbins]
        mu_pred = amp * torch.exp(-0.5 * ((bins - mean) / std)**2)

        # 3. Subtract predicted mu to get the residual signal
        residual_signal = data - mu_pred
        
        # 4. Predict the epsilon (noise) from the residual signal
        epsilon_pred = self.epsilon(residual_signal)
        
        # 5. Calculate loss
        mask = ( x['ni'] != 0 )  
        squared_error = (epsilon_pred - epsilon_sim)**2                                             # [B, N_bins]
        l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
        return (l * mask.float()).sum() * 0.5
