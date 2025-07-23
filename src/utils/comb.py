import numpy as np
from scipy.ndimage import uniform_filter1d
from collections import defaultdict
import torch

class Comb3:
    def __init__(self, waveform, timespace, ampscale=10**2, inject=True):
        assert len(waveform) == len(timespace), 'Waveform and Waveform Time Domain must be same length!!'
        
        self.asc = ampscale
        self.length = len(waveform)

        self.w_TD_data = waveform
        self.w_TD_times = timespace

        if inject:
            self.inject_comb_stoch(waveform,timespace)
        else:
            self.load_FD(waveform,timespace)

    def load_FD(self, wf,ts):
        self.w_FD_data = np.fft.rfft(wf)
        self.w_FD_freqs = np.fft.rfftfreq(len(wf), d=np.mean(np.diff(ts)))
        self.w_FD_df = np.mean(np.diff(ts))
    
    def _inject_comb(self,wf,ts, f0,df,nf):
        global base
        base = np.zeros(len(self.w_FD_data))
        comb_freqs = np.arange(nf)*df+f0
        global indices
        indices = np.argmin(np.abs(self.w_FD_freqs[:, np.newaxis] - comb_freqs[np.newaxis, :]), axis=0)
        
        smooth = uniform_filter1d(np.abs(np.real(self.w_FD_data)), size=10)

        # f_values = np.abs(np.real(self.w_FD_data[indices]))
        f_values = smooth[indices]

        base[indices] =  np.random.normal(loc=f_values*self.asc, scale=1, size=np.shape(f_values))

        self.i_FD_comb = base
        self.i_FD_data = base + self.w_FD_data
        self.ni = base!=0

        self.i_TD_data = np.fft.irfft(self.i_FD_data)
        self.i_TD_data_comb = np.fft.irfft(base)
    
    def inject_comb_known(self,wf,ts, f0,df,nf):
        self.load_FD(wf,ts)
        self._inject_comb(wf,ts, f0,df,nf)
        

    def inject_comb_stoch(self,wf,ts):
        self.load_FD(wf,ts)

        res = 10*self.w_FD_df
        nf = np.random.randint(2,10)
        f0 = np.random.uniform(low=10*res,high=(np.max(np.real(self.w_FD_freqs))-10*res))
        df = np.random.uniform(low=res, 
                               high=((np.max(np.real(self.w_FD_freqs))-f0)/nf))
        
        print(f'min {f0}, max {np.max(np.real(self.w_FD_freqs))}')
        # print(f'high {((np.max(np.real(self.w_FD_freqs))-f0)/nf)}')
        # print(f'Generating stochastic comb res {res}, nf {nf}, f0 {f0}, df {df}')
        self._inject_comb(wf,ts, f0,df,nf)

    def _sample(self) -> dict:
        self.inject_comb_stoch(wf=self.w_TD_data,ts=self.w_TD_times)
        sample = {}

        x0 = self.w_FD_data
        ni = self.ni
        epsilon = self.i_FD_comb
        xi = self.i_FD_data

        sample.update({'x0': x0, 'epsilon': epsilon, 'ni': ni, 'xi': xi})
        return sample
    
    def sample(self, Nsims):
        samples_list = [self._sample() for _ in range(Nsims)]
        samples_dict = defaultdict(list)
        for d in samples_list:
            for key, value in d.items():
                samples_dict[key].append(value)
        for key, value in d.items():
            samples_dict[key] = np.stack(samples_dict[key])
        return samples_dict
    
class Comb3_Torch:
    def __init__(self, waveform, timespace, ampscale=10**2, inject=True, device='cpu'):
        assert len(waveform) == len(timespace), 'Waveform and Waveform Time Domain must be same length!!'
        
        self.device = device
        self.asc = ampscale
        self.length = len(waveform)

        self.w_TD_data = torch.as_tensor(waveform, dtype=torch.float32).to(self.device)
        self.w_TD_times = torch.as_tensor(timespace, dtype=torch.float32).to(self.device)

        if inject:
            self.inject_comb_stoch(self.w_TD_data, self.w_TD_times)
        else:
            self.load_FD(self.w_TD_data, self.w_TD_times)

    def load_FD(self, wf, ts):
        self.w_FD_data = torch.fft.rfft(wf)
        d = torch.mean(torch.diff(ts))
        self.w_FD_freqs = torch.fft.rfftfreq(len(wf), d=d.item()).to(self.device)
        self.w_FD_df = d
    
    def _inject_comb(self, wf, ts, f0, df, nf):
        base = torch.zeros_like(self.w_FD_data)
        comb_freqs = torch.arange(nf, device=self.device, dtype=torch.float32) * df + f0
        indices = torch.argmin(torch.abs(self.w_FD_freqs[:, None] - comb_freqs[None, :]), axis=0)
        
        # PyTorch equivalent of uniform_filter1d
        smooth_size = 10
        abs_real_w_FD_data = torch.abs(torch.real(self.w_FD_data)).view(1, 1, -1)
        # Reflect padding for even kernel size
        padding = (smooth_size // 2, smooth_size // 2 - (1 if smooth_size % 2 == 0 else 0))
        padded_data = F.pad(abs_real_w_FD_data, padding, mode='reflect')
        kernel = torch.ones(1, 1, smooth_size, device=self.device) / smooth_size
        smooth = F.conv1d(padded_data, kernel).view(-1)

        f_values = smooth[indices]

        means = f_values * self.asc
        stds = torch.ones_like(means)
        
        noise = torch.normal(mean=means, std=stds)
        
        base[indices] = noise.to(base.dtype)

        self.i_FD_comb = base
        self.i_FD_data = base + self.w_FD_data
        self.ni = base != 0

        self.i_TD_data = torch.fft.irfft(self.i_FD_data)
        self.i_TD_data_comb = torch.fft.irfft(base)
    
    def inject_comb_known(self, wf, ts, f0, df, nf):
        self.load_FD(wf, ts)
        self._inject_comb(wf, ts, f0, df, nf)
        
    def inject_comb_stoch(self, wf, ts):
        self.load_FD(wf, ts)

        res = 10 * self.w_FD_df
        nf = torch.randint(2, 10, (1,), device=self.device).item()
        
        max_freq = torch.max(torch.real(self.w_FD_freqs))
        f0_low = 10 * res
        f0_high = max_freq - 10 * res
        
        if f0_low.item() >= f0_high.item():
            f0_low = max_freq / 4
            f0_high = max_freq / 2

        f0 = (f0_high - f0_low) * torch.rand(1, device=self.device) + f0_low
        
        df_low = res
        df_high = (max_freq - f0) / nf
        
        if df_low.item() >= df_high.item():
            df_low = res
            df_high = res * 2

        df = (df_high - df_low) * torch.rand(1, device=self.device) + df_low
        
        print(f'min {f0.item()}, max {max_freq.item()}')
        self._inject_comb(wf, ts, f0.item(), df.item(), nf)

    def _sample(self, wf, ts) -> dict:
        self.inject_comb_stoch(wf=wf, ts=ts)
        sample = {}

        x0 = self.w_FD_data
        ni = self.ni
        epsilon = self.i_FD_comb
        xi = self.i_FD_data

        sample.update({'x0': x0, 'epsilon': epsilon, 'ni': ni, 'xi': xi})
        return sample
    
    def sample(self, wf,ts, Nsims):
        samples_list = [self._sample(wf,ts) for _ in range(Nsims)]
        samples_dict = defaultdict(list)
        for d in samples_list:
            for key, value in d.items():
                samples_dict[key].append(value)
        for key, value in d.items():
            samples_dict[key] = torch.stack(samples_dict[key])
        return samples_dict
    

class Sim_FD_Additive:
    def __init__(self, Nbins, sigma, PSD_arr:torch.Tensor, bounds=5, fraction=None, sample_fraction=False, 
                 white = True, device='cpu', dtype=torch.float64):
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
        self.device = device
        self.dtype = dtype
        self.Nbins = Nbins
        self.sigma = sigma
        self.sigbounds = bounds
        self.fraction = fraction
        self.sample_fraction = sample_fraction
        self.grid = torch.linspace(1, 1024, Nbins, device=device, dtype=dtype)
        self.white = white
        if not white:
            self.base_PSD = PSD_arr

    
    def get_mu(self) -> torch.Tensor:
        grid = self.grid.unsqueeze(0)
        if self.white:
            return torch.ones(grid.shape)
        else:
            return self.base_PSD
    
    def get_x_H0(self, Nsims: int, mu: torch.Tensor = 0) -> torch.Tensor:
        x_shape = (Nsims, self.Nbins)
        return torch.from_numpy(np.random.lognormal(mean=mu,sigma=torch.sqrt(mu),size=x_shape)).to(self.dtype)
    
    def get_ni(self, x: torch.Tensor) -> torch.Tensor:
        # if self.fraction is None:
        """Standard basis vectors"""
        batch_size, N_bins = x.shape
        ni = torch.zeros(batch_size, N_bins, device=self.device, dtype=self.dtype)
        indices = torch.randint(0, N_bins, (batch_size,), device=self.device)
        ni[torch.arange(batch_size), indices] = 1
        # else:
        #     """Fraction of bins are distorted"""
        #     if self.sample_fraction:
        #         fr = np.random.uniform(0.01, self.fraction)
        #     else:   
        #         fr = self.fraction
        #     prob = fr*self.Nbins/100
        #     random_vals = torch.rand_like(x)
        #     ni = (random_vals < prob).type(self.dtype)  # fr% chance
        return ni
    
    def get_bounds(self,x:torch.Tensor) ->torch.Tensor:
        up = torch.exp(x+self.sigbounds*torch.sqrt(x))
        down = torch.exp(x-self.sigbounds*torch.sqrt(x))
        return up, down 

    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        up, down = self.get_bounds(x)
        # print(f'The Shape of the bounds follows {up.shape}, {down.shape}')
        ### THIS IS SUB OPTIMAL ###
        x_shape = np.shape(x.numpy)
        eps_np = np.random.uniform(low=down, high=up)
        return torch.from_numpy(eps_np)
        # return torch.FloatTensor(x.shape).uniform_(down, up)*ni
    
    def get_x_Hi(self, epsilon: torch.Tensor, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + epsilon * ni
    
    def _sample(self, Nsims: int) -> dict:
        sample = {}
        mu = self.get_mu()
        sample['mu'] = mu
        x0 = self.get_x_H0(Nsims, mu)
        ni = self.get_ni(x0)
        epsilon = self.get_epsilon(ni, x0)
        xi = self.get_x_Hi(epsilon, ni, x0)
        sample.update({'x0': x0, 'epsilon': epsilon, 'ni': ni, 'xi': xi})
        return sample
    
    def _resample(self, sample: dict) -> dict:
        Nsims = sample['x0'].shape[0] if sample['x0'].ndim == 2 else 1
        sample['x0'] = self.get_x_H0(Nsims, sample['mu'])
        sample['ni'] = self.get_ni(sample['x0'])
        sample['epsilon'] = self.get_epsilon(sample['ni'], sample['x0'])
        sample['xi'] = self.get_x_Hi(sample['epsilon'], sample['ni'], sample['x0'])
        return sample
    
    def sample(self, Nsims: int = 1) -> dict:
        sample = self._sample(Nsims)
        return sample