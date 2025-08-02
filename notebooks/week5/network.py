import torch
from models.online_norm import OnlineStandardizingLayer
from models.resnet_1d import ResidualNet

class Network_epsilon(torch.nn.Module):
    def __init__(self, Nbins):
        super().__init__()
        
        # A single, asymmetric variance model for the network's likelihood
        self.logvariance_left = torch.nn.Parameter(torch.ones(Nbins) * 5)
        self.logvariance_right = torch.nn.Parameter(torch.ones(Nbins) * 5)

        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        # MLP to predict the 3 parameters of the Gaussian mu signal
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
        # Use the average variance for a general SNR calculation
        avg_variance = (self.logvariance_left.exp() + self.logvariance_right.exp()) / 2
        return self.epsilon(x) / (avg_variance.sqrt() + 1e-10)

    def bounds(self):
        # Use the average standard deviation for generating the symmetric simulated noise
        avg_std = (self.logvariance_left.exp().sqrt() + self.logvariance_right.exp().sqrt()) / 2
        return avg_std.mean(-1) * 5
        
    def bounds_asym(self):
        """Returns the left and right bounds for asymmetric noise generation."""
        left_std = self.logvariance_left.exp().sqrt()
        right_std = self.logvariance_right.exp().sqrt()
        left_bound = left_std.mean(-1) * 5
        right_bound = right_std.mean(-1) * 5
        return left_bound, right_bound

    def forward(self, x):
        
        # Adaptive data generation
        ni = x['ni']
        
        ###########################################
        # Generate noise from an asymmetric uniform distribution
        left_bound, right_bound = self.bounds_asym()
        rand_val = torch.rand_like(x['x0'], device=x['x0'].device, dtype=x['x0'].dtype)
        
        # Sample from U[-left_bound, right_bound]
        epsilon_sim = ((left_bound + right_bound) * rand_val - left_bound) * ni
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
        
        # 5. Calculate loss using the single asymmetric likelihood
        mask = ( x['ni'] != 0 ).float()
        error = epsilon_pred - epsilon_sim
        
        # Determine which side of the distribution the error falls on
        is_left = (error > 0)
        
        # Select the appropriate variance and log-variance for each element
        variances = torch.where(is_left, self.logvariance_left.exp(), self.logvariance_right.exp())
        logvariances = torch.where(is_left, self.logvariance_left, self.logvariance_right)
        
        # Calculate the asymmetric likelihood
        squared_error = error**2
        l = squared_error / (variances + 1e-10) + logvariances
        
        return (l * mask).sum() * 0.5