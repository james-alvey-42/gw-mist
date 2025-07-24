import numpy as np
import torch
import sys
sys.path.append('../../mist-base/GW')
import gw150814_simulator as gs
    
class Simulator_Additive:
    def __init__(self, Nbins, sigma, bounds=5, fraction=None, 
                 sample_fraction=False, bkg=False, device='cpu', 
                 dtype=torch.float64, mode=None):
        """
        Args:
        - Nbins (int): Number of bins in the histogram.
        - sigma (float): Standard deviation of the Gaussian noise.
        - bounds (float): Bounds for the uniform distribution of the additive noise.
        - fraction (float): Fraction of bins to be perturbed by the additive noise. If None, just one bin is perturbed.
        - bkg (bool): If True, the simulator generates background events.
        - device (str): Device to run the tensors on.
        - dtype (torch.dtype): Data type of the tensors.
        """
        self.mode = mode
        if self.mode == 'gw':
            default = gs.defaults
            default['posterior_samples_path'] = '../../mist-base/GW/GW150814_posterior_samples.npz'
            default['f_max']=250
            self.gw = gs.GW150814(settings=default)
            self.Nbins = len(self.gw.time_to_frequency_domain(self.gw.generate_time_domain_waveform()))
        else: 
            self.Nbins = Nbins
        
        self.device = device
        self.dtype = dtype
        self.sigma = sigma
        self.bounds = bounds
        self.bkg = bkg
        self.fraction = fraction
        self.sample_fraction = sample_fraction
        self.grid = torch.linspace(20, 1024, self.Nbins, device=device, dtype=dtype)

        # print(f'self.Nbins {self.Nbins}')
        # print(f'shape of self.grid {self.grid.shape}')


            
        
    def get_theta(self, Nsims: int) -> torch.Tensor:
        return torch.rand(Nsims, 3, device=self.device, dtype=self.dtype) * 2 - 1

    def get_mu(self, theta: torch.Tensor) -> torch.Tensor:
        grid = self.grid.unsqueeze(0)  # Shape: (1, Nbins)
        # Compute mu for all simulations at once
        mu = (
            torch.sin(grid + 0.5*theta[:, 0:1]) +
            theta[:, 1:2] * grid/10 +
            0.5*theta[:, 2:3]
        )
        # return mu  # Shape: (Nsims, Nbins)
        # print(f'the shape of mu is {grid.shape}')
        return torch.zeros_like(grid)
    
    def get_x_H0(self, Nsims: int, mu: torch.Tensor = 0) -> torch.Tensor:
        x_shape = (Nsims, self.Nbins)
        if self.mode == 'white':
            noise = (torch.randn(x_shape, device=self.device, dtype=self.dtype) * self.sigma).to(self.dtype)
            return mu + noise
        elif self.mode == 'complex':
            noise = torch.complex(torch.rand(x_shape), torch.rand(x_shape)).to(self.dtype)
            norm_noise = torch.abs(noise)
            return mu+norm_noise
        elif self.mode == 'gw':
            gwsim = self.gw
            wf_td = gwsim.generate_time_domain_waveform()
            noise_td = gwsim.generate_time_domain_noise()

            N = len(wf_td)
            fs = 1/gwsim.delta_t

            wf_PSD = (2.0 / (fs * N)) * np.abs(gwsim.time_to_frequency_domain(wf_td))**2
            noise_PSD = (2.0 / (fs * N)) * np.abs(gwsim.time_to_frequency_domain(noise_td))**2

            whitened_bkg = torch.from_numpy(np.expand_dims((wf_PSD/noise_PSD),0))
            return mu+whitened_bkg
        else:
            raise Exception('pick a valid mode- white, gw or complex') 

    def get_ni(self, x: torch.Tensor) -> torch.Tensor:
        if self.fraction is None:
            """Standard basis vectors"""
            batch_size, N_bins = x.shape
            ni = torch.zeros(batch_size, N_bins, device=self.device, dtype=self.dtype)
            indices = torch.randint(0, N_bins, (batch_size,), device=self.device)
            ni[torch.arange(batch_size), indices] = 1
        else:
            """Fraction of bins are distorted"""
            if self.sample_fraction:
                fr = np.random.uniform(0.01, self.fraction)
            else:   
                fr = self.fraction
            # print(f'{self.fraction},{self.sample_fraction}')
            # print(f'simulating with a fraction {fr}')
            # prob = fr*self.Nbins/100
            prob = fr
            # print(f'prob {prob}')
            random_vals = torch.rand_like(x)
            ni = (random_vals < prob).type(self.dtype)  # fr% chance
        return ni
    
    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (2 * self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) - self.bounds) * ni
    
    def get_x_Hi(self, epsilon: torch.Tensor, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + epsilon * ni
    
    def _sample(self, Nsims: int) -> dict:
        sample = {}
        if self.bkg:
            theta = self.get_theta(Nsims)
            mu = self.get_mu(theta)
            sample['theta'] = theta
            sample['mu'] = mu
            x0 = self.get_x_H0(Nsims, mu)
        else:
            x0 = self.get_x_H0(Nsims, 0)
        ni = self.get_ni(x0)
        epsilon = self.get_epsilon(ni, x0)
        xi = self.get_x_Hi(epsilon, ni, x0)
        sample.update({'x0': x0, 'epsilon': epsilon, 'ni': ni, 'xi': xi})
        return sample
    
    def _resample(self, sample: dict) -> dict:
        Nsims = sample['x0'].shape[0] if sample['x0'].ndim == 2 else 1
        if self.bkg:
            sample['x0'] = self.get_x_H0(Nsims, sample['mu'])
        else:  
            sample['x0'] = self.get_x_H0(Nsims, 0)
        sample['ni'] = self.get_ni(sample['x0'])
        sample['epsilon'] = self.get_epsilon(sample['ni'], sample['x0'])
        sample['xi'] = self.get_x_Hi(sample['epsilon'], sample['ni'], sample['x0'])
        return sample
    
    def sample(self, Nsims: int = 1) -> dict:
        sample = self._sample(Nsims)
        return sample