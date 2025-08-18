import numpy as np
import torch
import sys
import jax
import jax.numpy as jnp
sys.path.append('../../mist-base/GW')
sys.path.append('mist-base/GW')
import gw150814_simulator as gs

class Simulator_Additive:
    def __init__(self, Nbins, sigma, bounds=5, fraction=None, 
                 sample_fraction=False, bkg=False, device='cpu', 
                 dtype=torch.float64, mode=None, bump = None,
                 specific_theta = None, frange = [None,None],
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
        """
        self.mode = mode
        
        # CHANGE START: Lazy Initialization for Multiprocessing
        # To prevent pickling errors when using num_workers > 0 in a DataLoader,
        # the complex gs.GW150814 object is no longer created here.
        # Instead, it will be initialized on-demand in each worker process.
        self.gw = None
        self._gw_settings = None
        # CHANGE END

        if self.mode == 'gw':
            # CHANGE START: Store settings for deferred initialization
            self._gw_settings = gs.defaults
            self._gw_settings['posterior_samples_path'] = '../../mist-base/GW/GW150814_posterior_samples.npz'
            # Attributes that depend on the gw object are now set in _init_gw()
            self.frange = [20,1024] if (frange[0] == None) else frange
            # CHANGE END
        else: 
            self.Nbins = Nbins
            self.grid = torch.linspace(0, self.Nbins, self.Nbins, device=device, dtype=dtype)
            self.frange = [0,self.Nbins] if (frange[0] == None) else frange
            self.mask = (self.grid>=self.frange[0])&(self.grid<=self.frange[1])
            self.grid_chopped = self.grid[self.mask]

        self.device = device
        self.dtype = dtype
        self.sigma = sigma
        self.bounds = np.abs(bounds)
        self.bkg = bkg
        self.fraction = fraction
        self.sample_fraction = sample_fraction
        self.bump = bump
        self.lock_amp = lock_amp
        self.lock_mu = lock_mu
        self.lock_sigma = lock_sigma
        self.spec_theta = specific_theta

    # CHANGE START: New method for lazy initialization
    def _init_gw(self):
        """
        Initializes the GW simulator object and its dependent attributes.
        This method is called on the first use within a process, ensuring that
        the un-pickleable gs.GW150814 object is created in the worker process
        itself, rather than being pickled by the main process.
        """
        if self.gw is None and self.mode == 'gw':
            self.gw = gs.GW150814(settings=self._gw_settings)
            self.Nbins = len(self.gw.frequencies)
            self.mask = (self.gw.frequencies >= self.frange[0]) & (self.gw.frequencies < self.frange[1])
            self.grid = self.gw.frequencies
            self.grid_chopped = self.grid[self.mask]
            self.psdnorm = torch.tensor(np.sqrt(self.gw.psd))
    # CHANGE END

    ########### GW SETUP STUFF #################

    def _fd_noise(self,nsims):
        # CHANGE: Ensure GW object is initialized before use
        self._init_gw()
        xshape = [nsims, len(self.grid)]
        white_noise_fd = (
            np.random.normal(size=xshape)
            + 1j * np.random.normal(size=xshape)
        ) / np.sqrt(2)
        prefactor = np.sqrt(self.gw.psd) / np.sqrt(2 * self.gw.delta_f)
        noise_fd = prefactor * white_noise_fd
        torch_mask = torch.from_numpy(np.array(self.mask))
        return (torch.tensor(np.abs(noise_fd))/self.psdnorm)[:,torch_mask]
    
    def _fd_theta_batched(self,nsims):
        # CHANGE: Ensure GW object is initialized before use
        self._init_gw()
        if self.bump != 'stoch':
            choices = np.random.choice(self.gw.posterior_array.shape[0], size=1, replace=True)
            single_param_set = self.gw.posterior_array[choices]
            params_batch = np.tile(single_param_set, (nsims, 1))
            return torch.from_numpy(params_batch)
        else:
            choices = np.random.choice(self.gw.posterior_array.shape[0], size=nsims, replace=True)
            params_batch = self.gw.posterior_array[choices]
            return torch.from_numpy(params_batch)

    def _fd_waveform_batched(self, params_batch_tensor):
        # CHANGE: Ensure GW object is initialized before use
        self._init_gw()
        params_batch = params_batch_tensor.numpy()
        theta_ripple_batch = params_batch[:, :8]
        ra_batch, dec_batch, psi_batch = params_batch[:, 8], params_batch[:, 9], params_batch[:, 10]
        batched_waveform = jax.vmap(self.gw.call_waveform)
        batched_detector_response = jax.vmap(
            self.gw.detector.fd_response,
            in_axes=(None, {'p': 0, 'c': 0}, {'ra': 0, 'dec': 0, 'psi': 0, 'gmst': None})
        )
        hp_batch, hc_batch = batched_waveform(theta_ripple_batch)
        wf_fd_batch = batched_detector_response(
            self.gw.frequencies,
            {"p": hp_batch, "c": hc_batch},
            {"ra": ra_batch, "dec": dec_batch, "psi": psi_batch, "gmst": self.gw.gmst},
        )
        wf_fd_block = torch.from_numpy(np.array(wf_fd_batch))
        torch_mask = torch.from_numpy(np.array(self.mask))
        return (torch.abs(wf_fd_block) / self.psdnorm)[:,torch_mask]
    
    ############### STOCHAISTIC GAUSSIAN SETUP STUFF ############

    def _gauss(self, x: torch.Tensor, m, amp, sigma) -> torch.Tensor:
        return amp * np.exp(-0.5 * ((x - m) / sigma) ** 2)
    
    def _gauss_theta_batched(self,nsims):
        theta_default = torch.Tensor([self.Nbins/2,3,self.Nbins/24]) if (self.spec_theta == None) else self.spec_theta
        theta_locked = theta_default*torch.ones(nsims, 3)
        if self.bump != 'stoch':
            return theta_locked
        else:
            norm = torch.tensor([self.Nbins/5,1,8])
            start = torch.tensor([self.Nbins/2, 3,self.Nbins/24])
            theta = torch.abs(torch.rand(nsims, 3, device=self.device, dtype=self.dtype) * norm + start)
            locks = torch.tensor([self.lock_mu, self.lock_amp, self.lock_sigma], device=self.device, dtype=torch.bool)
            output = torch.where(locks, theta_locked, theta)
            return output
    
    def _gauss_mu_batched(self,nsims,theta:torch.Tensor):
        grid = self.grid*torch.ones([nsims,len(self.grid)])
        mu = self._gauss(grid, theta[:,0].unsqueeze(-1), theta[:,1].unsqueeze(-1), theta[:,2].unsqueeze(-1))
        torch_mask = torch.as_tensor(self.mask)
        return (mu)[:,torch_mask]
    
    ######### GET COMMANDS ############

    def get_theta(self, Nsims: int) -> torch.Tensor:
        if self.mode == 'gw':
            # CHANGE: Ensure GW object is initialized before use
            self._init_gw()
            return self._fd_theta_batched(nsims=Nsims)
        else:
            return self._gauss_theta_batched(nsims=Nsims)

    def get_mu(self, Theta: torch.Tensor) -> torch.Tensor:
        Nsims = Theta.shape[0]
        if self.mode == 'gw':
            # CHANGE: Ensure GW object is initialized before use
            self._init_gw()
            return self._fd_waveform_batched(Theta)
        else:
            return self._gauss_mu_batched(Nsims, Theta)
    
    def get_noise_H0(self, Nsims):
        if self.mode == 'gw':
            # CHANGE: Ensure GW object is initialized before use
            self._init_gw()
        x_shape = (Nsims, len(self.grid_chopped))
        if self.mode == 'gw':
            return self._fd_noise(nsims=Nsims)
        elif self.mode == 'complex':
            noise = torch.complex(torch.randn(x_shape), torch.randn(x_shape))
            return torch.abs(noise).to(self.dtype)
        else:
            return (torch.randn(x_shape, device=self.device, dtype=self.dtype) * self.sigma).to(self.dtype)

    def get_x_H0(self, Nsims: int, mu: torch.Tensor = 0) -> torch.Tensor:
        return mu + self.get_noise_H0(Nsims=Nsims)

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
            prob = fr
            random_vals = torch.rand_like(x)
            ni = (random_vals < prob).type(self.dtype)
        return ni
    
    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (2 * self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) - self.bounds) * ni
    
    def get_x_Hi(self, epsilon: torch.Tensor, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + epsilon * ni
    
    def _sample(self, Nsims: int) -> dict:
        if self.mode == 'gw':
            # CHANGE: Ensure GW object is initialized before use
            self._init_gw()
        sample = {}
        x_shape = (Nsims, len(self.grid_chopped))
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