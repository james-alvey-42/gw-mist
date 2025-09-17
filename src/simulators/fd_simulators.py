import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey
import jax
import jax.numpy as jnp
from astropy.time import Time
from ripple.waveforms import IMRPhenomD, IMRPhenomXAS
from jimgw.single_event.detector import H1, L1
from functools import partial
import sys
import plotfancy as pf
import time

sys.path.append('../../mist-base/GW')
sys.path.append('../../')

import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps


defaults = {
    "approximant": "IMRPhenomD",
    "f_min": 20., #40.0,
    "f_max": 1024., #400.0,
    "f_ref": 20.0,
    "notches": [60.0, 120.0, 180.0],
    "notch_width": 0.1,
    "trigger_time": 1126259462.4,
    "pre_trigger_time": 2.0,
    "post_trigger_time": 2.0,
    "data_window": 1000,
    "psd_window":4,
    # "psd_pad": 16.0,
    "ifo": "H1",
    # "tukey_alpha": 0.2,
    "posterior_samples_path": "GW150814_posterior_samples.npz",
}

class Base_GW1501914:
    def __init__(self, settings:dict={}, stoch_mu=True):
        self.on = False
        self.settings = settings
        self.stoch = stoch_mu
        
        self._init_all()

    ### METHODS ###

    def _init_all(self):
        if not self.on:
            self.on = True 
            self._unpack_settings()
            self._initialise_gw()
            self._load_posterior_samples()
            self._setup_waveform()
    
    def _unpack_settings(self):
        self.approximant = self.settings.get("approximant", "IMRPhenomD")
        self.f_min = self.settings.get("f_min", 20.)
        self.f_max = self.settings.get("f_max", 1024.)
        self.f_ref = self.settings.get("f_ref", 20.0)
        self.notches = self.settings.get("notches", [60.0, 120.0, 180.0])
        self.notch_width = self.settings.get("notch_width", 0.1)
        self.trigger_time = self.settings.get("trigger_time", 1126259462.4)
        self.pre_trigger_time = self.settings.get("pre_trigger_time", 2.0)
        self.post_trigger_time = self.settings.get("post_trigger_time", 2.0)
        self.data_window = self.settings.get("data_window", 1000)
        self.psd_window = self.settings.get("psd_window", 2)
        self.ifo = self.settings.get("ifo", "H1")
        self.posterior_samples_path = self.settings.get(
            "posterior_samples_path", "../../mist-base/GW/GW150814_posterior_samples.npz"
        )

    def _initialise_gw(self):
        self.event_time = event_gps('GW150914')
        # print(f'Fetching {self.data_window}s of data from {self.ifo}. Will take <4mins.')
        self.gwosc_data = TimeSeries.fetch_open_data(self.ifo, self.event_time-2*self.data_window, 
                                                     self.event_time-self.data_window, sample_rate=4096)
        # print(f'Calculating PSD for FFT window length {self.psd_window}s.')
        self.psd = self.gwosc_data.psd(fftlength=self.psd_window)
        self.frequencies = self.psd.frequencies.value
        self.tukey_alpha = 1/self.psd_window

        self.times = self.gwosc_data.times.value - self.trigger_time
        self.duration = self.gwosc_data.duration.value
        self.npts = len(self.gwosc_data)
        self.delta_t = self.gwosc_data.dt.value
        self.epoch = self.duration - self.post_trigger_time
        self.gmst = (
            Time(self.trigger_time, format="gps")
            .sidereal_time("apparent", "greenwich")
            .rad
        )
        if self.ifo == "H1":
            self.detector = H1
        elif self.ifo == "L1":
            self.detector = L1
        
    def _tukey(self, x):
        # x is supplied in form [nsamples,nbins]
        x = np.array(x)
        window = tukey(np.shape(x)[1], alpha=self.tukey_alpha) # shape [nbins]
        return x*np.expand_dims(window,axis=0) # shape [nsamples, nbins]
    
    def whiten(self, x):
        # x is supplied in form [nbins (complex)]
        x = np.array(x)
        return x/np.sqrt(self.psd.data)
    
    def whiten_batch(self,x):
        # x is supplied in form [nsamples,nbins]
        x = np.array(x)
        prefactor = np.expand_dims(np.sqrt(np.array(self.psd.data)), axis=0)
        return x/prefactor
         
    
    # def _load_posterior_samples(self):
    #     # print(f"Loading posterior samples from {self.posterior_samples_path}")
    #     self.posterior_samples = np.load(self.posterior_samples_path)
    #     self.parameter_names = [
    #         "M_c",
    #         "q",
    #         "s1_z",
    #         "s2_z",
    #         "d_L",
    #         "t_c",
    #         "phase_c",
    #         "iota",
    #         "ra",
    #         "dec",
    #         "psi",
    #     ]
    #     for param_name in self.posterior_samples.files:
    #         if param_name not in self.parameter_names:
    #             raise ValueError(
    #                 f"Parameter {param_name} not recognized in posterior file"
    #             )
    #     self.posterior_array = np.vstack(
    #         [self.posterior_samples[name] for name in self.parameter_names]
    #     ).T
    #     self.posterior_array[:, 1] = (self.posterior_array[:, 1]) / (
    #         1.0 + self.posterior_array[:, 1]
    #     ) ** 2  # q -> eta = q / (1 + q)^2
    #     self.posterior_array[:, 5] = (
    #         self.epoch + self.posterior_array[:, 5]
    #     )  # t_c -> t_c + epoch

    def _load_posterior_samples(self):
        # print(f"Loading posterior samples from {self.posterior_samples_path}")
        with np.load(self.posterior_samples_path) as f:
            self.posterior_samples = {key: f[key] for key in f.files}
            self.parameter_names = [
                "M_c",
                "q",
                "s1_z",
                "s2_z",
                "d_L",
                "t_c",
                "phase_c",
                "iota",
                "ra",
                "dec",
                "psi",
            ]
            for param_name in self.posterior_samples.keys():
                if param_name not in self.parameter_names:
                    raise ValueError(
                        f"Parameter {param_name} not recognized in posterior file"
                    )
        self.posterior_array = np.vstack(
            [self.posterior_samples[name] for name in self.parameter_names]
        ).T
        self.posterior_array[:, 1] = (self.posterior_array[:, 1]) / (
            1.0 + self.posterior_array[:, 1]
        ) ** 2  # q -> eta = q / (1 + q)^2
        self.posterior_array[:, 5] = (
            self.epoch + self.posterior_array[:, 5]
        )  # t_c -> t_c + epoch

    def _setup_waveform(self):
        if self.approximant == "IMRPhenomD":
            self.waveform = IMRPhenomD.gen_IMRPhenomD_hphc
        elif self.approximant == "IMRPhenomXAS":
            self.waveform = IMRPhenomXAS.gen_IMRPhenomXAS_hphc
        else:
            raise ValueError(f"Approximant {self.approximant} not recognized")
    
    ### DATA GETTERS ###

    def _get_noise_td(self, nsamples):
        ref = self.event_time-1.5*self.data_window
        nbins = len(self.gwosc_data.crop(ref-self.psd_window/2, ref+self.psd_window/2))
        alldata_raw = jnp.array(self.gwosc_data)
        max_start_index = alldata_raw.shape[0] - nbins
        start_indices = jnp.array(np.random.randint(0, max_start_index + 1, size=nsamples))
        offsets = jnp.arange(nbins)
        indices = start_indices[:, None] + offsets
        return alldata_raw[indices]
    
    def get_noise_fd(self, nsamples):
        noise_td = self._get_noise_td(nsamples=nsamples)
        noise_td_windowed = self._tukey(noise_td)
        noise_fd_complex = jnp.fft.rfft(noise_td_windowed)*self.gwosc_data.dt
        return self.whiten_batch(noise_fd_complex)
    
    def get_GW150914_td(self):
        return TimeSeries.fetch_open_data(self.ifo, self.event_time-self.psd_window/2, 
                                                     self.event_time+self.psd_window/2, sample_rate=4096)
        
    def get_GW150914_fd(self):
        raw = self.get_GW150914_td().data
        raw_windowed = np.squeeze(self._tukey(np.expand_dims(raw,axis=0)), axis=0)
        raw_fd = np.fft.rfft(raw_windowed)*self.gwosc_data.dt
        return self.whiten(raw_fd)

    ### WAVEFORM GENERATORS ###

    @partial(jax.jit, static_argnums=(0,))
    def call_waveform(self, theta_ripple):
        hp, hc = self.waveform(self.frequencies, theta_ripple, f_ref=self.f_ref)
        return hp, hc
    
    def _fd_theta_batched(self,nsims):
        if not self.stoch:
            choices = np.random.choice(self.posterior_array.shape[0], size=1, replace=True)
            single_param_set = self.posterior_array[choices]
            params_batch = np.tile(single_param_set, (nsims, 1))
            return torch.from_numpy(params_batch)
        else:
            choices = np.random.choice(self.posterior_array.shape[0], size=nsims, replace=True)
            params_batch = self.posterior_array[choices]
            return torch.from_numpy(params_batch)
        
    def _fd_waveform_batched(self, params_batch_tensor):
        params_batch = params_batch_tensor.numpy()
        theta_ripple_batch = params_batch[:, :8]
        ra_batch, dec_batch, psi_batch = params_batch[:, 8], params_batch[:, 9], params_batch[:, 10]
        batched_waveform = jax.vmap(self.call_waveform)
        batched_detector_response = jax.vmap(
            self.detector.fd_response,
            in_axes=(None, {'p': 0, 'c': 0}, {'ra': 0, 'dec': 0, 'psi': 0, 'gmst': None})
        )
        hp_batch, hc_batch = batched_waveform(theta_ripple_batch)
        wf_fd_batch = batched_detector_response(
            self.frequencies,
            {"p": hp_batch, "c": hc_batch},
            {"ra": ra_batch, "dec": dec_batch, "psi": psi_batch, "gmst": self.gmst},
        )
        return self.whiten_batch(wf_fd_batch)
    
    


class GW_Additive_F(Base_GW1501914):
    def __init__(self, settings: dict = {}, stoch_mu=True, device='cpu', dtype=torch.float64, 
                 bkg=True,bounds=5,fraction=None,sample_fraction=None):
        
        super().__init__(settings, stoch_mu)

        self.device = device
        self.dtype = dtype
        self.bkg = bkg
        self.bounds = bounds
        self.fraction = fraction
        self.sample_fraction = sample_fraction

        self.Nbins = len(self.frequencies)
        self.grid = self.frequencies

        self.dtype_map = {
            torch.float64:torch.complex128,
            torch.float32:torch.complex64,
            torch.float16:torch.complex32
        }
        self.complex_dtype = self.dtype_map.get(self.dtype,torch.complex128)

    def to_type(self, tensr:torch.Tensor) -> torch.Tensor:
        return tensr.to(dtype=self.dtype, device=self.device)
    
    def to_type_complex(self, tensr:torch.Tensor) -> torch.Tensor:
        complex_dtype = self.dtype_map.get(self.dtype,torch.complex128)
        return tensr.to(dtype=complex_dtype, device=self.device)

    def get_theta(self, Nsims: int) -> torch.Tensor:
        self._init_all()
        if self.bkg:
            output = self._fd_theta_batched(Nsims)
            return self.to_type(output)
        else:
            output = torch.zeros([Nsims, self.Nbins])
            return self.to_type(output)
    
    def get_mu(self, theta: torch.Tensor) -> torch.Tensor:
        self._init_all()
        mu = torch.tensor(self._fd_waveform_batched(theta))
        return self.to_type_complex(mu)
    
    def get_x_H0(self, m:torch.Tensor,n:torch.Tensor, mag=True) -> torch.Tensor:
        self._init_all()
        return m+n
    
    # def get_ni(self, x: torch.Tensor) -> torch.Tensor:
    #     self._init_all()
    #     # xreal = torch.abs(torch.tensor(x)).squeeze(0)
    #     xreal = torch.abs(x)
    #     if self.fraction is None:
    #         """Standard basis vectors"""
    #         batch_size, N_bins = xreal.shape
    #         ni = torch.zeros(batch_size, N_bins, device=self.device, dtype=self.dtype)
    #         indices = torch.randint(0, N_bins, (batch_size,), device=self.device)
    #         ni[torch.arange(batch_size), indices] = 1
    #     else:
    #         """Fraction of bins are distorted"""
    #         if self.sample_fraction:
    #             fr = np.random.uniform(0.01, self.fraction)
    #         else:   
    #             fr = self.fraction
    #         prob = fr*self.Nbins/100
    #         random_vals = torch.rand_like(xreal)
    #         ni = (random_vals < prob).type(self.dtype)  # fr% chance
    #     return ni
    
    def get_ni(self, x: torch.Tensor, real:bool=True) -> torch.Tensor:
        dt = self.dtype if real else self.complex_dtype
        self._init_all()
        # xreal = torch.abs(torch.tensor(x)).squeeze(0)
        xreal = torch.abs(x)
        if self.fraction is None:
            """Standard basis vectors"""
            batch_size, N_bins = xreal.shape
            ni = torch.zeros(batch_size, N_bins, device=self.device, dtype=dt)
            indices = torch.randint(0, N_bins, (batch_size,), device=self.device)
            ni[torch.arange(batch_size), indices] = 1
        else:
            """Fraction of bins are distorted"""
            if self.sample_fraction:
                fr = np.random.uniform(0.01, self.fraction)
            else:   
                fr = self.fraction
            prob = fr*self.Nbins/100
            random_vals = torch.rand_like(xreal)
            ni = (random_vals < prob).type(dt)  # fr% chance
        return ni

    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor, real:bool=True) -> torch.Tensor:
        self._init_all()
        # xreal = torch.abs(torch.tensor(x)).squeeze(0)
        xreal = torch.abs(x)
        if real:
            return (2 * self.bounds * torch.rand(xreal.shape, device=self.device, dtype=self.complex_dtype).real - self.bounds) * ni
        else:
            return (2 * self.bounds * torch.rand(xreal.shape, device=self.device, dtype=self.complex_dtype) - self.bounds) * ni
    
    def get_x_Hi(self, epsilon: torch.Tensor, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._init_all()
        return x + epsilon * ni
    
    def get_x_Hi_real(self,epsilon: torch.Tensor, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._init_all()
        return torch.abs(x) + epsilon.real*ni

    def _sample(self, Nsims: int, Real:bool=True) -> dict:
        self._init_all()
        sample = {} 

        Theta = self.get_theta(Nsims) ## note i have built in the bkg method to theta to make this cleaner
        Mu = self.get_mu(Theta)
        # X0_c = self.get_x_H0(Mu)
        # X0 = torch.abs(X0_c)
        Noise = torch.tensor(self.get_noise_fd(Mu.shape[0]))
        X0 = self.get_x_H0(Mu, Noise)
        Ni = self.get_ni(X0, real=Real)
        Epsilon = self.get_epsilon(Ni, X0, real=Real)
        Xi = self.get_x_Hi(Epsilon, Ni, X0)
        Xi_r = self.get_x_Hi_real(Epsilon, Ni, X0)

        # sample.update({'theta':Theta,'mu':Mu, 'x0': X0, 'x0_c':X0_c,
        #                'epsilon': Epsilon, 'ni': Ni, 'xi': Xi})
        
        sample.update({'theta':Theta,'mu':Mu,'noise':Noise, 'x0': X0,
                       'epsilon': Epsilon, 'ni': Ni, 'xi': Xi, 'xi_r':Xi_r})
    
        return sample
    
    def _resample(self, sample: dict) -> dict:
        self._init_all()
        sample['x0'] = self.get_x_H0(sample['mu'])
        sample['ni'] = self.get_ni(sample['x0'])
        sample['epsilon'] = self.get_epsilon(sample['ni'], sample['x0'])
        sample['xi'] = self.get_x_Hi(sample['epsilon'], sample['ni'], sample['x0'])
        return sample
    
    def sample(self, Nsims: int = 1, REAL:bool=True) -> dict:
        sample = self._sample(Nsims,)
        return sample



class GW_Additive_F_Correlated(GW_Additive_F):
    def __init__(self, settings: dict = {}, stoch_mu=True, device='cpu', 
                 dtype=torch.float64, bkg=True, bounds=5, fraction=None, 
                 sample_fraction=None, 
                 correlation_scales = 2**torch.linspace(0, 5, 6).int()):
        super().__init__(settings, stoch_mu, device, dtype, bkg, bounds, 
                         fraction, sample_fraction)
        
        self.correlation_scales = correlation_scales

    def _conv1d(self, ni: torch.Tensor, c: int) -> torch.Tensor:
        self._init_all()
        w = torch.linspace(-3, 3, 1+int(c)*2, device=ni.device, dtype=ni.dtype).unsqueeze(0).unsqueeze(0)
        w = torch.exp(-0.5*w**2)
        w = w/w.max() # Normalize maximum to 1
        y = torch.nn.functional.conv1d(ni.unsqueeze(1), w.to(ni.dtype), padding = int(c)).squeeze(1)
        return y
    
    def get_correlation(self, ni, epsilon) -> torch.Tensor:
        self._init_all()
        if isinstance(self.bounds, float) or isinstance(self.bounds, int):
            cc = torch.stack([self._conv1d(ni*epsilon, int(c)) for c in self.correlation_scales], axis=1)
        elif len(self.bounds)==len(self.correlation_scales):
            cc = torch.stack([self._conv1d(ni*epsilon[:, i_c], int(c)) for i_c, c in enumerate(self.correlation_scales)], axis=1)
        return cc
    
    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._init_all()
        if isinstance(self.bounds, float) or isinstance(self.bounds, int):
            eps = (2 * self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) - self.bounds) * ni
        elif len(self.bounds)==len(self.correlation_scales):
            eps = (2 * self.bounds.unsqueeze(0).unsqueeze(2) * 
                   torch.rand(x.shape, device=self.device, dtype=self.dtype
                              ).unsqueeze(1).to(self.dtype) - self.bounds.unsqueeze(0).unsqueeze(2)
                              ) * ni.unsqueeze(1)
        return eps.to(self.dtype)
        
    def get_x_Hi(self, cni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._init_all()
        return x +  cni
    
    def _sample(self, Nsims: int) -> dict:
        self._init_all()
        sample = {}

        Theta = self.get_theta(Nsims)
        Mu = self.get_mu(Theta)
        X0_c = self.get_x_H0(Mu)
        X0 = torch.abs(X0_c)
        Ni = self.get_ni(X0)
        Epsilon = self.get_epsilon(Ni, X0)
        Cni = self.get_correlation(Ni, Epsilon)
        Xi = self.get_x_Hi(Cni, X0.unsqueeze(1))

        sample.update({'theta':Theta, 'mu':Mu, 'x0_c':X0_c,
                       'x0': X0, 'epsilon': Epsilon, 
                       'ni': Ni, 'cni': Cni, 'xi': Xi})
        return sample
