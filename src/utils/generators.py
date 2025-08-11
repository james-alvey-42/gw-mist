import numpy as np
import torch
import sys
sys.path.append('../../mist-base/GW')
import gw150814_simulator as gs
    
class Simulator_Additive:
    def __init__(self, Nbins, sigma, bounds=5, fraction=None, 
                 sample_fraction=False, bkg=False, device='cpu', 
                 dtype=torch.float64, mode=None, bump = None, pve_bounds =True,
                 specific_theta = None,
                 lock_amp = False, lock_mu=False, lock_sigma=False):
        """
        Args:
        - Nbins (int): Number of bins in the histogram.
        - sigma (float): Standard deviation of the Gaussian noise.
        - bounds (float): Bounds for the uniform distribution of the additive noise.
        - fraction (float): Fraction of bins to be perturbed by the additive noise. If None, just one bin is perturbed.
        - bkg (bool): If True, the simulator generates a bkground signal.
        - device (str): Device to run the tensors on.
        - dtype (torch.dtype): Data type of the tensors.
        - mode (str): white, complex or gw - simulation generation mode
        - bump (str or list): None, det or [a,m,s] - whether to generate a bump as mu and whether it is stochastic or not
        - pve_bounds (bool): If True, 
        """
        self.mode = mode
        if self.mode == 'gw':
            default = gs.defaults
            default['posterior_samples_path'] = '../../mist-base/GW/GW150814_posterior_samples.npz'
            self.gw = gs.GW150814(settings=default)
            self.Nbins = len(self.gw.time_to_frequency_domain(self.gw.generate_time_domain_waveform())[100:1024])
        else: 
            self.Nbins = Nbins
        # print(f'the number of bins is {self.Nbins}')

        self.device = device
        self.dtype = dtype
        self.sigma = sigma
        self.bounds = np.abs(bounds)
        self.bkg = bkg
        self.fraction = fraction
        self.sample_fraction = sample_fraction
        self.grid = torch.linspace(100, 1024, self.Nbins, device=device, dtype=dtype)
        self.bump = bump
        self.pve_bounds = pve_bounds

        self.lock_amp = lock_amp
        self.lock_mu = lock_mu
        self.lock_sigma = lock_sigma

        self.spec_theta = specific_theta

        # print(f'self.Nbins {self.Nbins}')
        # print(f'shape of self.grid {self.grid.shape}')

    def _gauss(self, x: torch.Tensor, m, amp, sigma) -> torch.Tensor:
        return amp * np.exp(-0.5 * ((x - m) / sigma) ** 2)
            
    def get_theta(self, Nsims: int) -> torch.Tensor:
        ##### NB YOU WILL NEED TO CHANGE THIS FOR THE GW METHOD - ONLY WORKS FOR GRID ON 0-100 HERE ####
        theta_locked = torch.tensor([self.Nbins/2,3,self.Nbins/24])*torch.ones(Nsims, 3)
        if self.bump != 'stoch':
            return theta_locked if self.spec_theta == None else self.spec_theta
        else:
            norm = torch.tensor([self.Nbins/5,1,0.01])
            start = torch.tensor([self.Nbins/2, 1,self.Nbins/24])
            theta = torch.rand(Nsims, 3, device=self.device, dtype=self.dtype) * norm + start
            locks = torch.tensor([self.lock_mu, self.lock_amp, self.lock_sigma], device=self.device, dtype=torch.bool)
            output = torch.where(locks, theta_locked, theta)
            return output


    def get_mu(self, theta: torch.Tensor) -> torch.Tensor:
        Nsims = theta.shape[0]
        base = torch.zeros(self.Nbins).unsqueeze(0)
        grid = torch.arange(self.Nbins).unsqueeze(0)*torch.ones([Nsims,self.Nbins])
        mu = self._gauss(grid, theta[:,0].unsqueeze(-1), theta[:,1].unsqueeze(-1), theta[:,2].unsqueeze(-1))
        # print(theta.shape)
        # print(mu.shape)
        return mu
    
    def get_x_H0(self, Nsims: int, mu: torch.Tensor = 0) -> torch.Tensor:
        x_shape = (Nsims, self.Nbins)
        if self.mode == 'white':
            noise = (torch.randn(x_shape, device=self.device, dtype=self.dtype) * self.sigma).to(self.dtype)
            return mu + noise
        elif self.mode == 'complex':
            noise = torch.complex(torch.randn(x_shape), torch.randn(x_shape))
            norm_noise = torch.abs(noise).to(self.dtype)
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
            h0 = mu+whitened_bkg[:,100:1024]
            # self.Nbins = h0.shape[1]
            return h0 ### APPLY CUT HERE TO MAKE SURE FILTER ISN"T INCLUDED
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
    
    # def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor: OLD - MODIFIED BELOW
    #     return (2 * self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) - self.bounds) * ni
    
    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.pve_bounds:
            return self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) * ni # returns on [0, self.bounds)
        else:
            return (2 * self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) - self.bounds) * ni # returns on [-self.bounds, self.bounds)
    
    def get_x_Hi(self, epsilon: torch.Tensor, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + epsilon * ni
    
    def _sample(self, Nsims: int) -> dict:
        sample = {}
        x_shape = (Nsims, self.Nbins)
        if self.bkg:
            theta = self.get_theta(Nsims)
            mu = self.get_mu(theta)
            sample['theta'] = theta
            sample['mu'] = mu
            x0 = self.get_x_H0(Nsims, mu)
        else:
            mu = torch.zeros(x_shape).to(dtype=self.dtype)
            sample['mu'] = mu.to(dtype=self.dtype)
            x0 = self.get_x_H0(Nsims, 0).to(dtype=self.dtype)
        ni = self.get_ni(x0).to(dtype=self.dtype)
        epsilon = self.get_epsilon(ni, x0).to(dtype=self.dtype)
        xi = self.get_x_Hi(epsilon, ni, x0).to(dtype=self.dtype)
        
        sample.update({'x0': x0,'epsilon': epsilon, 'ni': ni, 'xi': xi})
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