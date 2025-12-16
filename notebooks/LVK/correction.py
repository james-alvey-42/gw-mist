import numpy as np
import bilby
import h5py
import matplotlib.pyplot as plt
import tqdm

ifos = [] 
h_white_waveform = {ifo.name: [] for ifo in ifos}
# --- End Placeholder Initializations ---


model = 'SEOBNRv5PHM'
archive = h5py.File('posterior_samples.h5')
posterior = archive.get('C00:'+model)['posterior_samples']


if model == "SEOBNRv5PHM":
    fd_model = bilby.gw.source.gwsignal_binary_black_hole
    waveform_arguments = {}
elif model == "IMRPhenomXPHM":
    fd_model = bilby.gw.source.lal_binary_black_hole
    waveform_arguments = {'PhenomXHMReleaseVersion': 122022, 'PhenomXPFinalSpinMod': 2, 'PhenomXPrecVersion': 320}
else:
    fd_model = bilby.gw.source.lal_binary_black_hole
    waveform_arguments = {}

waveform_arguments.update(
    {
        "waveform_approximant": model,
        "minimum_frequency": 10,
        "maximum_frequency": 448.0,
        "reference_frequency": 10,
    }
)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=8.0, sampling_frequency=1024.0,
    frequency_domain_source_model=fd_model,
    start_time=1384782888.634277 - 6.,
    waveform_arguments=waveform_arguments
)

# CORRECTION: The 'posterior' object is an HDF5 Dataset, which acts like a
# structured NumPy array. It does not have a .items() method. To iterate,
# we first get the names of the parameters from its dtype.
parameter_names = posterior.dtype.names
n_samples = len(posterior)
inds = np.arange(n_samples)

for ii in tqdm.tqdm(inds):
    # Then, for each sample 'ii', we build the params dictionary by accessing
    # the i-th value from each parameter column in the dataset.
    params = {key: posterior[key][ii] for key in parameter_names}
    
    pols = waveform_generator.frequency_domain_strain(parameters=params)
    for ifo in ifos:
        h = ifo.get_detector_response(pols, params)
        frequency_window_factor = (
            np.sum(ifo.frequency_mask)
            / len(ifo.frequency_mask)
        )
        ht = h / (ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4))
        h_white_waveform[ifo.name].append(
            np.fft.irfft(ht)
            * np.sqrt(np.sum(ifo.frequency_mask)) / frequency_window_factor
        )
