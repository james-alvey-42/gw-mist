import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
import numpy as np
from ripple.waveforms import IMRPhenomD, IMRPhenomXAS
from jimgw.single_event.detector import H1, L1
from functools import partial
from astropy.time import Time
from tqdm import tqdm
import torch
from collections import defaultdict


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
    "psd_window": 16.0,
    "psd_pad": 16.0,
    "ifo": "H1",
    "tukey_alpha": 0.2,
    "posterior_samples_path": "GW150814_posterior_samples.npz",
}


class GW150814:
    def __init__(self, settings={}):
        self.settings = settings
        self.unpack_settings()
        self.load_data()
        self.load_psd()
        self.load_posterior_samples()
        self.setup_fourier_transform()
        self.setup_filter()
        self.setup_waveform()

    def unpack_settings(self):
        self.approximant = self.settings.get("approximant", "IMRPhenomD")
        self.f_min = self.settings.get("f_min", 20.) #40.0)
        self.f_max = self.settings.get("f_max", 1024.) #400.0)
        self.f_ref = self.settings.get("f_ref", 20.0)
        self.notches = self.settings.get("notches", [60.0, 120.0, 180.0])
        self.notch_width = self.settings.get("notch_width", 0.1)
        self.trigger_time = self.settings.get("trigger_time", 1126259462.4)
        self.pre_trigger_time = self.settings.get("pre_trigger_time", 2.0)
        self.post_trigger_time = self.settings.get("post_trigger_time", 2.0)
        self.psd_window = self.settings.get("psd_window", 16.0)
        self.psd_pad = self.settings.get("psd_pad", 16.0)
        self.ifo = self.settings.get("ifo", "H1")
        self.tukey_alpha = self.settings.get("tukey_alpha", 0.2)
        self.posterior_samples_path = self.settings.get(
            "posterior_samples_path", "GW150814_posterior_samples.npz"
        )

    def load_data(self):
        print(f"Loading data for {self.ifo} at GPS time {self.trigger_time}")
        self.data_gwosc = TimeSeries.fetch_open_data(
            self.ifo,
            self.trigger_time - self.pre_trigger_time,
            self.trigger_time + self.post_trigger_time,
        )
        self.times = self.data_gwosc.times.value - self.trigger_time
        self.duration = self.data_gwosc.duration.value
        self.npts = len(self.data_gwosc)
        self.delta_t = self.data_gwosc.dt.value
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

    def load_psd(self):
        print(f"Loading PSD for {self.ifo} at GPS time {self.trigger_time} [can take O(1 min)]")
        start_psd = (
            self.trigger_time - self.pre_trigger_time - self.psd_pad - self.psd_window
        )
        end_psd = self.trigger_time - self.pre_trigger_time - self.psd_pad
        psd_data_td = TimeSeries.fetch_open_data(self.ifo, int(start_psd), int(end_psd))
        self.psd = psd_data_td.psd(fftlength=self.duration).value

    def load_posterior_samples(self):
        print(f"Loading posterior samples from {self.posterior_samples_path}")
        self.posterior_samples = np.load(self.posterior_samples_path)
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
        for param_name in self.posterior_samples.files:
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

    def setup_fourier_transform(self):
        self.frequencies = jnp.fft.rfftfreq(self.npts, self.delta_t)
        self.delta_f = self.frequencies[1] - self.frequencies[0]

    def setup_filter(self):
        bandpass = jnp.logical_and(
            self.frequencies > self.f_min, self.frequencies < self.f_max
        )
        notches = jnp.ones_like(self.frequencies)
        for notch in self.notches:
            notches = jnp.logical_and(
                notches,
                jnp.logical_or(
                    self.frequencies < notch - self.notch_width,
                    self.frequencies > notch + self.notch_width,
                ),
            )
        self.filter = jnp.logical_and(bandpass, notches)

    def setup_waveform(self):
        if self.approximant == "IMRPhenomD":
            self.waveform = IMRPhenomD.gen_IMRPhenomD_hphc
        elif self.approximant == "IMRPhenomXAS":
            self.waveform = IMRPhenomXAS.gen_IMRPhenomXAS_hphc
        else:
            raise ValueError(f"Approximant {self.approximant} not recognized")

    @partial(jax.jit, static_argnums=(0,))
    def call_waveform(self, theta_ripple):
        hp, hc = self.waveform(self.frequencies, theta_ripple, f_ref=self.f_ref)
        return hp, hc

    def time_to_frequency_domain(self, data):
        return (
            jnp.fft.rfft(jnp.array(data) * tukey(self.npts, self.tukey_alpha))
            * self.delta_t
        )

    def frequency_to_time_domain(self, data):
        return jnp.fft.irfft(jnp.array(data)) / self.delta_t
    
    def whitened_frequency_to_time_domain(self, data):
        return jnp.fft.irfft( jnp.array(data) / jnp.sqrt(self.psd) ) / self.delta_t

    def filter_gwosc_data(self):
        data_gwosc_fd = self.time_to_frequency_domain(self.data_gwosc.value)
        # self.data_gwosc_filtered = self.frequency_to_time_domain(
        #     data_gwosc_fd * self.filter
        # )
        self.data_gwosc_filtered = self.whitened_frequency_to_time_domain(
            data_gwosc_fd * self.filter
        )
        return self.data_gwosc_filtered

    def generate_time_domain_noise(self):
        white_noise_fd = (
            np.random.normal(size=len(self.psd))
            + 1j * np.random.normal(size=len(self.psd))
        ) / np.sqrt(2)
        prefactor = np.sqrt(self.psd) / np.sqrt(2 * self.delta_f)
        noise_fd = prefactor * white_noise_fd
        # noise_td = self.frequency_to_time_domain(noise_fd * self.filter)
        noise_td = self.whitened_frequency_to_time_domain(noise_fd * self.filter)
        return noise_td

    def generate_time_domain_waveform(self):
        choice = np.random.choice(self.posterior_array.shape[0])
        params = self.posterior_array[choice]
        theta_ripple = jnp.array(
            [
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6],
                params[7],
            ]
        )  # M_c, eta = q / (1 + q)^2, s1_z, s2_z, d_L, t_c + epoch, phase_c, iota
        ra, dec, psi = params[8], params[9], params[10]
        hp, hc = self.call_waveform(theta_ripple)
        hdet_fd = self.detector.fd_response(
            self.frequencies,
            {"p": hp, "c": hc},
            params={"ra": ra, "dec": dec, "psi": psi, "gmst": self.gmst},
        )
        # return self.frequency_to_time_domain(hdet_fd * self.filter)
        return self.whitened_frequency_to_time_domain(hdet_fd * self.filter)
    
    def downsample_time_series(self, data, factor):
        if len(data) % factor != 0:
            raise ValueError(f"Data length {len(data)} is not divisible by factor {factor}")
        return np.mean(data.reshape(-1, factor), axis=1)
    
    def get_td_mask(self, tmin, tmax):
        return np.logical_and(self.times >= tmin, self.times <= tmax)

    ##############################
    def _process(self, data):
        downsampled_data = self.downsample_time_series(data, 8)
        downsampled_time = self.downsample_time_series(self.times, 8)
        mask = np.logical_and(downsampled_time >= -0.1, downsampled_time <= 0.1)
        donwsampled_masked_data = downsampled_data[mask]
        return donwsampled_masked_data
        
    def _jax_to_torch(self, data):
        return torch.from_numpy(np.asarray(data))
    
    def _sample(self) -> dict:
        mu = self.generate_time_domain_waveform()
        noise = self.generate_time_domain_noise()
        # Checks
        if jnp.isnan(mu.max()):
            print('Nan in waveform data')
        elif jnp.isnan(noise.max()):
            print('Nan in noise data')
            
        mu = self._process(mu)
        noise = self._process(noise)
        # Checks
        if jnp.isnan(mu.max()):
            print('Nan in donwsampled waveform data')
        elif jnp.isnan(noise.max()):
            print('Nan in donwsampled noise data')
            
        mu = self._jax_to_torch(mu)
        noise = self._jax_to_torch(noise)
        # Checks
        if torch.isnan(mu.max()):
            print('Nan in waveform data')
        elif torch.isnan(noise.max()):
            print('Nan in noise data')
        
        return {'mu': mu, 'noise': noise}
    
    def sample(self, Nsims=1, show_progress=True):
        if show_progress:
            samples_list = [self._sample() for _ in tqdm(range(Nsims))]
        else:  
            samples_list = [self._sample() for _ in range(Nsims)]
        samples_dict = defaultdict(list)
        for d in samples_list:
            for key, value in d.items():
                samples_dict[key].append(value)
        for key, value in d.items():
            samples_dict[key] = torch.stack(samples_dict[key])
        return samples_dict
    
    
    
class GW150814_Additive(GW150814):
    def __init__(self, gw150814_samples, settings={}, correlation_scales=None, bounds=1, sigma=None, fraction=None, sample_fraction=False, device='cpu', dtype=torch.float64):
        super().__init__(settings)
        
        self.gw150814_samples = gw150814_samples
        self.device = device
        self.dtype = dtype
        self.bounds = bounds
        self.fraction = fraction
        self.sample_fraction = sample_fraction
        self.correlation_scales = correlation_scales
        self.sigma = sigma # For gaussian noise
        
        # Create separate random number generators
        seed_mu = 0
        seed_noise = 42
        seed_distortion = 3
        self.mu_rng = np.random.default_rng(seed_mu)
        self.noise_rng = np.random.default_rng(seed_noise)
        self.distortion_rng = np.random.default_rng(seed_distortion)
        self.mu_torch_rng = torch.Generator(device=device)
        self.mu_torch_rng.manual_seed(seed_mu)
        self.noise_torch_rng = torch.Generator(device=device)
        self.noise_torch_rng.manual_seed(seed_noise)
        self.distortion_torch_rng = torch.Generator(device=device)
        self.distortion_torch_rng.manual_seed(seed_distortion)
        
    @property
    def Nbins(self):
        return self.gw150814_samples['mu'].shape[1]
        
    def get_mu(self) -> torch.Tensor:
        idx = self.mu_rng.integers(0, self.gw150814_samples['mu'].shape[0])
        return (self.gw150814_samples['mu'][idx]).to(self.device, self.dtype) #* 1e21
        # return torch.zeros_like(self.gw150814_samples['mu'][idx]).to(self.device, self.dtype)
    
    def get_x_H0(self, mu: torch.Tensor = 0) -> torch.Tensor:
        if self.sigma is None:
            idx = self.noise_rng.integers(0, self.gw150814_samples['noise'].shape[0])
            noise = (self.gw150814_samples['noise'][idx]).to(self.device, self.dtype) #  * 1e21
        else:
            noise = torch.randn(mu.shape, generator=self.noise_torch_rng, device=self.device, dtype=self.dtype) * self.sigma
        return mu + noise
    
    def get_ni(self, x: torch.Tensor) -> torch.Tensor:
        if self.fraction is None:
            """Standard basis vectors"""
            ni = torch.zeros(self.Nbins, device=self.device, dtype=self.dtype)
            indices = torch.randint(0, self.Nbins, (1,), generator=self.distortion_torch_rng)
            ni[indices] = 1
        else:
            """Fraction of bins are distorted"""
            if self.sample_fraction:
                fr = self.distortion_rng.uniform(0.01, self.fraction)
            else:   
                fr = self.fraction
            prob = fr*self.Nbins/100
            random_vals = torch.rand(x.shape, generator=self.distortion_torch_rng, device=self.device, dtype=self.dtype)
            ni = (random_vals < prob).type(self.dtype)  # fr% chance
        return ni
    
    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.bounds, float) or isinstance(self.bounds, int):
            return (2 * self.bounds * torch.rand(x.shape, generator=self.distortion_torch_rng, device=self.device, dtype=self.dtype) - self.bounds) * ni
        elif self.correlation_scales is not None and len(self.bounds)==len(self.correlation_scales):
            return (2 * self.bounds.unsqueeze(1) * torch.rand(x.shape, generator=self.distortion_torch_rng, device=self.device, dtype=self.dtype).unsqueeze(0) - self.bounds.unsqueeze(1)) * ni.unsqueeze(0)
    
    def _conv1d(self, ni: torch.Tensor, c: int) -> torch.Tensor:
        w = torch.linspace(-3, 3, 1+int(c)*2, device=ni.device, dtype=ni.dtype).unsqueeze(0).unsqueeze(0)
        w = torch.exp(-0.5*w**2)
        w = w/w.max() # Normalize maximum to 1
        if len(ni.shape)==1:
            y = torch.nn.functional.conv1d(ni.unsqueeze(0).unsqueeze(0), w.to(ni.dtype), padding = int(c)).squeeze(0).squeeze(0)
        elif len(ni.shape)==2:
            y = torch.nn.functional.conv1d(ni.unsqueeze(1), w.to(ni.dtype), padding = int(c)).squeeze(1)
        return y
    
    def get_correlation(self, ni, epsilon) -> torch.Tensor:
        if self.correlation_scales is None:
            return epsilon*ni
        if isinstance(self.bounds, float) or isinstance(self.bounds, int):
            cc = torch.stack([self._conv1d(ni*epsilon, int(c)) for c in self.correlation_scales], axis=0)
        elif len(self.bounds)==len(self.correlation_scales):
            cc = torch.stack([self._conv1d(ni*epsilon[i_c], int(cor)) for i_c, cor in enumerate(self.correlation_scales)], axis=0)
        return cc
    
    def get_x_Hi(self, x0: torch.Tensor, cni: torch.Tensor) -> torch.Tensor:
        return x0 + cni
    
    def _sample(self) -> dict:
        mu = self.get_mu()
        x0 = self.get_x_H0(mu)
        ni = self.get_ni(x0)
        epsilon = self.get_epsilon(ni, x0)
        cni = self.get_correlation(ni, epsilon)
        xi = self.get_x_Hi(x0, cni)
        return {'mu': mu, 'x0': x0, 'epsilon': epsilon, 'ni': ni, 'cni':cni, 'xi': xi}
    
    def _resample(self, sample: dict) -> dict: 
        sample['x0'] = self.get_x_H0(sample['mu'])
        sample['ni'] = self.get_ni(sample['x0'])
        sample['epsilon'] = self.get_epsilon(sample['ni'], sample['x0'])
        sample['cni'] = self.get_correlation(sample['ni'], sample['epsilon'])
        sample['xi'] = self.get_x_Hi(sample['x0'], sample['cni'])
        return sample
    
    def sample(self, Nsims=1, show_progress=False):
        if show_progress:
            samples_list = [self._sample() for _ in tqdm(range(Nsims))]
        else:  
            samples_list = [self._sample() for _ in range(Nsims)]
        samples_dict = defaultdict(list)
        for d in samples_list:
            for key, value in d.items():
                samples_dict[key].append(value)
        for key, value in d.items():
            samples_dict[key] = torch.stack(samples_dict[key])
        return samples_dict



# class GW150814_Additive_f(GW150814):
#     def __init__(self, gw150814_samples, settings={}, correlation_scales=None, bounds=1, sigma=None, fraction=None, sample_fraction=False, device='cpu', dtype=torch.float64):
#         super().__init__(settings)
        
#         self.gw150814_samples = gw150814_samples
#         self.device = device
#         self.dtype = dtype
#         self.bounds = bounds
#         self.fraction = fraction
#         self.sample_fraction = sample_fraction
#         self.correlation_scales = correlation_scales
#         self.sigma = sigma # For gaussian noise
        
#         # Create separate random number generators
#         seed_mu = 0
#         seed_noise = 42
#         seed_distortion = 3
#         self.mu_rng = np.random.default_rng(seed_mu)
#         self.noise_rng = np.random.default_rng(seed_noise)
#         self.distortion_rng = np.random.default_rng(seed_distortion)
#         self.mu_torch_rng = torch.Generator(device=device)
#         self.mu_torch_rng.manual_seed(seed_mu)
#         self.noise_torch_rng = torch.Generator(device=device)
#         self.noise_torch_rng.manual_seed(seed_noise)
#         self.distortion_torch_rng = torch.Generator(device=device)
#         self.distortion_torch_rng.manual_seed(seed_distortion)
        
#     @property
#     def Nbins(self):
#         return self.gw150814_samples['mu'].shape[1]
    
#     @property
#     def Nbins_OG(self):
#         return 16384
        
#     def get_mu(self) -> torch.Tensor:
#         idx = self.mu_rng.integers(0, self.gw150814_samples['mu'].shape[0])
#         return (self.gw150814_samples['mu'][idx] * 1e21).to(self.device, self.dtype)
    
#     def get_x_H0(self, mu: torch.Tensor = 0) -> torch.Tensor:
#         if self.sigma is None:
#             # Generate noise from the PSD on the fly
#             # white_noise_fd = (
#             #     np.random.normal(size=len(self.psd))
#             #     + 1j * np.random.normal(size=len(self.psd))
#             # ) / np.sqrt(2)
#             # prefactor = np.sqrt(self.psd) / np.sqrt(2 * self.delta_f)
#             # noise_fd = prefactor * white_noise_fd
#             # noise_td = self.frequency_to_time_domain(noise_fd * self.filter)
#             # noise = self._jax_to_torch(self._process(noise_td)).to(self.dtype) *1e21
#             # Generate noise from the PSD from store
#             idx = self.noise_rng.integers(0, self.gw150814_samples['noise'].shape[0])
#             noise = (self.gw150814_samples['noise'][idx] * 1e21).to(self.device, self.dtype)
#         else:
#             noise = torch.randn(mu.shape, generator=self.noise_torch_rng, device=self.device, dtype=self.dtype) * self.sigma
#         return mu + noise
    
#     def get_ni(self, x: torch.Tensor) -> torch.Tensor:
#         if self.fraction is None:
#             """Standard basis vectors"""
#             ni = torch.zeros(self.Nbins_OG, device=self.device, dtype=self.dtype)
#             indices = torch.randint(7790, 8600, (1,), generator=self.distortion_torch_rng)
#             ni[indices] = 1
#         else:
#             raise ValueError('Fractional distortion not implemented for frequency domain')
#         return ni
    
#     def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         if isinstance(self.bounds, float) or isinstance(self.bounds, int):
#             return (2 * self.bounds * torch.rand(ni.shape, generator=self.distortion_torch_rng, device=self.device, dtype=self.dtype) - self.bounds) * ni
#         elif self.correlation_scales is not None and len(self.bounds)==len(self.correlation_scales):
#             return (2 * self.bounds.unsqueeze(1) * torch.rand(ni.shape, generator=self.distortion_torch_rng, device=self.device, dtype=self.dtype).unsqueeze(0) - self.bounds.unsqueeze(1)) * ni.unsqueeze(0)
    
#     def _conv1d(self, ni: torch.Tensor, c: int) -> torch.Tensor:
#         w = torch.linspace(-3, 3, 1+int(c)*2, device=ni.device, dtype=ni.dtype).unsqueeze(0).unsqueeze(0)
#         w = torch.exp(-0.5*w**2)
#         w = w/w.max() # Normalize maximum to 1
#         if len(ni.shape)==1:
#             y = torch.nn.functional.conv1d(ni.unsqueeze(0).unsqueeze(0), w.to(ni.dtype), padding = int(c)).squeeze(0).squeeze(0)
#         elif len(ni.shape)==2:
#             y = torch.nn.functional.conv1d(ni.unsqueeze(1), w.to(ni.dtype), padding = int(c)).squeeze(1)
#         return y
    
#     def get_correlation(self, ni, epsilon) -> torch.Tensor:
#         if self.correlation_scales is None:
#             return epsilon*ni
#         if isinstance(self.bounds, float) or isinstance(self.bounds, int):
#             cc = torch.stack([self._conv1d(ni*epsilon, int(c)) for c in self.correlation_scales], axis=0)
#         elif len(self.bounds)==len(self.correlation_scales):
#             cc = torch.stack([self._conv1d(ni*epsilon[i_c], int(cor)) for i_c, cor in enumerate(self.correlation_scales)], axis=0)
#         return cc
    
#     def _frequency_mask(self, cni):
#         cni_f = self.time_to_frequency_domain(cni)
#         cni_t = self.frequency_to_time_domain(cni_f * self.filter)
#         cni_t = self._jax_to_torch(self._process(cni_t[0])).to(self.device, self.dtype)
#         return cni_t
    
#     def get_x_Hi(self, x0: torch.Tensor, cni: torch.Tensor) -> torch.Tensor:
#         return x0 + cni
    
#     def _sample(self) -> dict:
#         mu = self.get_mu()
#         x0 = self.get_x_H0(mu)
#         ni_OG = self.get_ni(x0)
#         epsilon_OG = self.get_epsilon(ni_OG, x0)
#         cni_OG = self.get_correlation(ni_OG, epsilon_OG)
#         ni = self._frequency_mask(ni_OG.unsqueeze(0))
#         epsilon = self._frequency_mask(epsilon_OG)
#         cni = self._frequency_mask(cni_OG).unsqueeze(0)
#         xi = self.get_x_Hi(x0, cni)
#         return {'mu': mu, 'x0': x0, 'epsilon_OG': epsilon_OG, 'epsilon': epsilon, 'ni': ni, 'cni':cni, 'xi': xi}
    
#     def _resample(self, sample: dict) -> dict: 
#         sample['x0'] = self.get_x_H0(sample['mu'])
#         # sample['ni_OG'] = self.get_ni(sample['x0'])
#         ni_OG = self.get_ni(sample['x0'])
#         sample['epsilon_OG'] = self.get_epsilon(ni_OG, sample['x0'])
#         # sample['cni_OG'] = self.get_correlation(ni_OG, sample['epsilon_OG'])
#         cni_OG = self.get_correlation(ni_OG, sample['epsilon_OG'])
#         sample['ni'] = self._frequency_mask(ni_OG.unsqueeze(0))
#         sample['epsilon'] = self._frequency_mask(sample['epsilon_OG'])
#         sample['cni'] = self._frequency_mask(cni_OG)
#         sample['xi'] = self.get_x_Hi(sample['x0'], sample['cni'])
#         return sample
    
#     def sample(self, Nsims=1, show_progress=False):
#         if show_progress:
#             samples_list = [self._sample() for _ in tqdm(range(Nsims))]
#         else:  
#             samples_list = [self._sample() for _ in range(Nsims)]
#         samples_dict = defaultdict(list)
#         for d in samples_list:
#             for key, value in d.items():
#                 samples_dict[key].append(value)
#         for key, value in d.items():
#             samples_dict[key] = torch.stack(samples_dict[key])
#         return samples_dict