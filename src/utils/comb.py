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