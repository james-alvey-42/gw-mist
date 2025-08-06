import torch

# Placeholder for Nbins, as it was not defined in the provided code
Nbins = 512

# Placeholder for ResidualNet, as its definition was not provided.
# This is a dummy implementation to make the code runnable.
class ResidualNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_features, num_blocks, kernel_size, padding):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, hidden_features, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_features, out_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.net(x)

# The original Network class provided by the user, with minor cleanup for portability
class Network_Epsilon_MU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        self.mu_predictor = torch.nn.Sequential(
            torch.nn.Linear(Nbins, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3) # 3 params: amplitude, mean, std
        )

    def _autogauss(self, x: torch.Tensor, t) -> torch.Tensor:
        # x is expected to be of shape [B, C, L], we use L for grid size and B for batching
        grid = torch.arange(x.shape[-1], device=x.device).expand(x.shape[0], -1)
        # t is [B, 3]
        # t[:,0] is amplitude, t[:,1] is mean, t[:,2] is log_std
        amp = t[:, 0].unsqueeze(1)
        mean = t[:, 1].unsqueeze(1)
        std = torch.exp(t[:, 2].unsqueeze(1))
        return amp * torch.exp(-0.5 * ((grid - mean) / std) ** 2)
    
    def epsilon(self,x):
        theta_pred = self.mu_predictor(x.unsqueeze(1)).squeeze(1)
        mu_pred = self._autogauss(x.unsqueeze(1),theta_pred)
        x_red = x - mu_pred
        return self.net(x_red.unsqueeze(1)).squeeze(1) + mu_pred
    
    def snr(self, x):
        return self.epsilon(x) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        # Adaptive data generation
        ni = x['ni']
        
        ###########################################
        epsilon_sim =  (2 * self.bounds() * torch.rand(x['x'].shape, 
                                                           device= x['x'].device, 
                                                           dtype= x['x'].dtype) - self.bounds()) * ni
        ###########################################
        
        data =  x['x0'] + epsilon_sim * ni
        
        # data = x['x']
        epsilon = self.epsilon(data)
        mask = ( x['ni'] != 0 )  
        squared_error = (epsilon - epsilon_sim)**2                                                  # [B, N_bins]
        l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
        return (l * mask.float()).sum() * 0.5

# New network with bounded outputs for mu_predictor
class Network_Epsilon_MU_Bounded(torch.nn.Module):
    """
    A modified version of the Network_Epsilon_MU that enforces bounds
    on the parameters (amplitude, mean, std) predicted by the mu_predictor.
    This can help with training stability and ensure the learned parameters
    are within a physically meaningful range.
    """
    def __init__(self, amp_max=5.0, std_log_max=3.0):
        """
        Initializes the network.
        Args:
            amp_max (float): The maximum value for the amplitude parameter.
            std_log_max (float): The maximum absolute value for the log standard deviation.
                                 The std will be in range [exp(-std_log_max), exp(std_log_max)].
        """
        super().__init__()
        
        self.amp_max = amp_max
        self.std_log_max = std_log_max
        
        self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        self.mu_predictor = torch.nn.Sequential(
            torch.nn.Linear(Nbins, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3) # 3 params: raw_amplitude, raw_mean, raw_log_std
        )

    def _autogauss(self, x: torch.Tensor, t) -> torch.Tensor:
        # x is expected to be of shape [B, C, L], we use L for grid size and B for batching
        grid = torch.arange(x.shape[-1], device=x.device).expand(x.shape[0], -1)
        # t is [B, 3]
        # t[:,0] is amplitude, t[:,1] is mean, t[:,2] is log_std
        amp = t[:, 0].unsqueeze(1)
        mean = t[:, 1].unsqueeze(1)
        std = torch.exp(t[:, 2].unsqueeze(1))
        return amp * torch.exp(-0.5 * ((grid - mean) / std) ** 2)
    
    def epsilon(self,x):
        raw_theta_pred = self.mu_predictor(x.unsqueeze(1)).squeeze(1)
        
        # Apply transformations to bound the parameters
        # 1. Amplitude: (0, amp_max)
        amplitude = torch.sigmoid(raw_theta_pred[:, 0]) * self.amp_max
        
        # 2. Mean: (0, Nbins-1)
        mean = torch.sigmoid(raw_theta_pred[:, 1]) * (Nbins - 1)
        
        # 3. Log Standard Deviation: (-std_log_max, std_log_max)
        # This keeps std in (exp(-std_log_max), exp(std_log_max))
        log_std = torch.tanh(raw_theta_pred[:, 2]) * self.std_log_max
        
        theta_pred = torch.stack([amplitude, mean, log_std], dim=1)
        
        mu_pred = self._autogauss(x.unsqueeze(1), theta_pred)
        x_red = x - mu_pred
        return self.net(x_red.unsqueeze(1)).squeeze(1) + mu_pred
    
    def snr(self, x):
        return self.epsilon(x) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        # Adaptive data generation
        ni = x['ni']
        
        ###########################################
        epsilon_sim =  (2 * self.bounds() * torch.rand(x['x'].shape, 
                                                           device= x['x'].device, 
                                                           dtype= x['x'].dtype) - self.bounds()) * ni
        ###########################################
        
        data =  x['x0'] + epsilon_sim * ni
        
        # data = x['x']
        epsilon = self.epsilon(data)
        mask = ( x['ni'] != 0 )  
        squared_error = (epsilon - epsilon_sim)**2                                                  # [B, N_bins]
        l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
        return (l * mask.float()).sum() * 0.5
