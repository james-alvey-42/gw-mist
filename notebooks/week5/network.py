import torch
from models.online_norm import OnlineStandardizingLayer
from models.resnet_1d import ResidualNet

class Network_epsilon(torch.nn.Module):
    def __init__(self, Nbins):
        super().__init__()
        
        # Epsilon network part - standard symmetric variance
        self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
        
        # Asymmetric variance parameters for the mu signal
        self.logvariance_mu_left = torch.nn.Parameter(torch.ones(Nbins) * 5)
        self.logvariance_mu_right = torch.nn.Parameter(torch.ones(Nbins) * 5)

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
        
        data =  x['x0'] + epsilon_sim * ni
        
        # 1. Predict mu parameters from the data
        mu_params = self.mu_predictor(data)
        
        # Unpack parameters
        amp = mu_params[:, 0].unsqueeze(1)
        mean = mu_params[:, 1].unsqueeze(1)
        std = torch.exp(mu_params[:, 2]).unsqueeze(1)

        # 2. Reconstruct the predicted mu signal
        bins = self.bins.unsqueeze(0)
        mu_pred = amp * torch.exp(-0.5 * ((bins - mean) / std)**2)

        # 3. Subtract predicted mu to get the residual signal
        residual_signal = data - mu_pred
        
        # 4. Predict the epsilon (noise) from the residual signal
        epsilon_pred = self.epsilon(residual_signal)
        
        # 5. Calculate a composite loss
        mask = (ni != 0).float()
        
        # --- Asymmetric Mu Loss ---
        error_mu = mu_pred - x['x0']
        # Where prediction is greater than true, it's a "left" error relative to the true value
        is_left_mu = (error_mu > 0) 
        
        variances_mu = torch.where(is_left_mu, self.logvariance_mu_left.exp(), self.logvariance_mu_right.exp())
        logvariances_mu = torch.where(is_left_mu, self.logvariance_mu_left, self.logvariance_mu_right)
        
        l_mu = error_mu**2 / (variances_mu + 1e-10) + logvariances_mu
        loss_mu = (l_mu * mask).sum() * 0.5

        # --- Symmetric Epsilon Loss ---
        squared_error = (epsilon_pred - epsilon_sim)**2
        l_epsilon = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance
        loss_epsilon = (l_epsilon * mask).sum() * 0.5
        
        # --- Combine Losses ---
        alpha = 0.5
        total_loss = alpha * loss_mu + (1 - alpha) * loss_epsilon
        
        return total_loss