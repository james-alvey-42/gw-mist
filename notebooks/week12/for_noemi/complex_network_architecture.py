import torch
import os

import os, sys

sys.path.append("../../mist-base")
sys.path.append('../../')

from models.resnet_1d import ResidualNet

class Network_epsilon_complex(torch.nn.Module):
    def __init__(self, nbins):
        super().__init__()
        
        self.nbins = nbins
        self.logvariance = torch.nn.Parameter(torch.ones(self.nbins, dtype=torch.float64) * 5)
        
        self.net = ResidualNet(2, 2, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) #now takes 2 input channels (real, imag) and outputs 2 (complex epsilon)
        self.mu_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.nbins * 2, 128), # Input is concatenated real and imaginary parts
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.nbins * 2)  # Output is concatenated real and imaginary parts
        )

    def mu(self, x): #here we build in complex processing - x is complex
        x_real_imag = torch.cat([x.real, x.imag], dim=-1) # Shape: [B, 2 * nbins]
        out_real_imag = self.mu_predictor(x_real_imag) # Shape: [B, 2 * nbins]
        pred_real, pred_imag = torch.chunk(out_real_imag, 2, dim=-1) # Each is [B, nbins]
        return torch.complex(pred_real, pred_imag)
 
    
    def epsilon(self, x): #here we build in complex processing - x is complex
        resd = x - self.mu(x) #this is the mu subtraction step
        resd_channels = torch.stack([resd.real, resd.imag], dim=1)
        out_channels = self.net(resd_channels)
        return torch.complex(out_channels[:, 0, :], out_channels[:, 1, :])
    
    def snr(self, x):
        return torch.abs(self.epsilon(x)) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        ### NOTE this nan_to_num step just sets the nan value at f=0 to zero. I just cut it out here as it is difficult to do on the simulation end of things.
        x0_block = torch.nan_to_num(x['x0'], nan=0) # Expected to be complex
        mu_block = torch.nan_to_num(x['mu'], nan=0) # Expected to be complex
        ni = torch.nan_to_num(x['ni'], nan=0) # Expected to be complex
        
        rand_real = 2 * torch.rand(x0_block.shape, device=x0_block.device) - 1
        rand_imag = 2 * torch.rand(x0_block.shape, device=x0_block.device) - 1
        epsilon_sim_unit = torch.complex(rand_real, rand_imag)
        epsilon_sim = self.bounds()*epsilon_sim_unit*ni

        data = x0_block + epsilon_sim # data is complex
        
        # net evaluation_m
        net_mu = self.mu(data) # net_mu is complex
        
        # Loss for complex numbers is the squared magnitude of the difference
        error_mu = (torch.abs(net_mu - mu_block))**2
        
        l_mu = error_mu / (self.logvariance.exp() + 1e-10) + self.logvariance
        l_mu_return = l_mu.sum() * 0.5

        # net evaluation_e
        net_epsilon = self.epsilon(data) # NOTE the subtraction is built into the epsilon method here. Net epsilon is complex
        mask = ( ni != 0 )
        
        # Same as above? 
        squared_error_e = (torch.abs(net_epsilon - epsilon_sim))**2                                     # [B, N_bins]
        l_e = squared_error_e / (self.logvariance.exp() + 1e-21) + self.logvariance               # [B, N_bins]
        l_e_return = (l_e * mask.float()).sum() * 0.5
        
        # combine
        return l_mu_return + l_e_return