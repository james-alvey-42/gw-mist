#Script used to produce the median waveform and 90% uncertainty for each waveform model (used in Fig. 1)

import bilby
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import tqdm
import glob
import pickle
from utils import *

# Run IMRPhenomXO4a in igwn-py310-20250129
#models = ["IMRPhenomXO4a"]
# Run other models in igwn-py310-20241106
models = ["IMRPhenomXPHM", "IMRPhenomTPHM", "SEOBNRv5PHM", "NRSur7dq4"]

psd_data = read_result.psd["C00:NRSur7dq4"]

channel_dict = {
        "H1": "H1:GDS-CALIB_STRAIN_CLEAN_BAYESWAVE_S00",
        "L1": "L1:GDS-CALIB_STRAIN_CLEAN_AR"
    }

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
for ifo in ifos:
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
         frequency_array=psd_data[ifo.name][:,0],
         psd_array=psd_data[ifo.name][:,1]
    )
    ifo.maximum_frequency = 448.0
    ifo.minimum_frequency = 20.0
    ifo.duration = 8.0
    ifo.sampling_frequency = 1024
    _data = TimeSeries.get(channel=channel_dict[ifo.name],start=1384782882.634277, end=1384782890.634277, verbose=False, allow_tape=True)
    _data = _data.crop(start=1384782888.634277 - 6., end=1384782888.634277 + 2.)
    lal_timeseries = _data.to_lal()
    lal.ResampleREAL8TimeSeries(
        lal_timeseries, float(1 / ifo.sampling_frequency)
    )
    _data = TimeSeries(
        lal_timeseries.data.data,
        epoch=lal_timeseries.epoch,
        dt=lal_timeseries.deltaT,
    )
    ifo.set_strain_data_from_gwpy_timeseries(_data)
    ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
        prefix=f"recalib_{ifo.name}_",
        minimum_frequency=ifo.minimum_frequency,
        maximum_frequency=ifo.maximum_frequency,
        n_points=10,
    )

for model in models:
    h_white_waveform = {"H1": [], "L1": []}
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
    posterior = samples_map[model]
    maxL_ind = np.argmax(posterior["log_likelihood"])
    maxL_params = {key: item[maxL_ind] for key, item in posterior.items()}
    pols = waveform_generator.frequency_domain_strain(parameters=maxL_params)
    h = {}
    for ifo in ifos:
        h[ifo.name] = ifo.get_detector_response(pols, maxL_params)

    h_white_maxL = {}
    for ifo in ifos:
        frequency_window_factor = (
            np.sum(ifo.frequency_mask)
            / len(ifo.frequency_mask)
        )
        ht = h[ifo.name] / (ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4))
        h_white_maxL[ifo.name] = (
            np.fft.irfft(ht)
            * np.sqrt(np.sum(ifo.frequency_mask)) / frequency_window_factor
        )

    inds = np.arange(len(posterior["chirp_mass"]))
    for ii in tqdm.tqdm(inds):
        params = {key: item[ii] for key, item in posterior.items()}
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
    # save model data
    #with open(f"../../Results/Results_for_Fig1/{model}_whitened_td_waveform.pkl", "wb") as f:
    #    pickle.dump(h_white_waveform, f)

    # save model specific uncertainty in dat file
    data = np.array(
        [
            waveform_generator.time_array,
            np.percentile(h_white_waveform["H1"], 5, axis=0),
            np.percentile(h_white_waveform["H1"], 50, axis=0),
            np.percentile(h_white_waveform["H1"], 95, axis=0),
            waveform_generator.time_array,
            np.percentile(h_white_waveform["L1"], 5, axis=0),
            np.percentile(h_white_waveform["L1"], 50, axis=0),
            np.percentile(h_white_waveform["L1"], 95, axis=0),
        ]
    ).T
    np.savetxt(
        f"../../Results/Results_for_Fig1/median_{model}_waveform_with_90pct_uncertainty.dat", data, delimiter="\t", header="\t".join(
            ["H1_time", "H1_lower", "H1_median", "H1_upper", "L1_time", "L1_lower", "L1_median", "L1_upper"]
        )
    )
