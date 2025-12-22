#Script used to combine the median waveform and 90% uncertainty from various waveform models (output used in Fig. 1)

import pickle
import bilby
import numpy as np

h_white = {"H1": [], "L1": []}
for model in ["IMRPhenomXPHM", "IMRPhenomTPHM", "SEOBNRv5PHM", "NRSur7dq4", "IMRPhenomXO4a"]:
    with open(f"../../Results/{model}_whitened_td_waveform.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    for ifo in h_white.keys():
        h_white[ifo] += model_data[ifo]

waveform_generator = bilby.gw.WaveformGenerator(
    duration=8.0, sampling_frequency=1024.0,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    start_time=1384782888.634277 - 6.,
    waveform_arguments={
        "waveform_approximant": "NRSur7dq4",
        "minimum_frequency": 10,
        "maximum_frequency": 448.0,
        "reference_frequency": 10,
    }
)
data = np.array(
    [
        waveform_generator.time_array,
        np.percentile(h_white["H1"], 5, axis=0), 
        np.percentile(h_white["H1"], 50, axis=0),
        np.percentile(h_white["H1"], 95, axis=0),
        waveform_generator.time_array,
        np.percentile(h_white["L1"], 5, axis=0), 
        np.percentile(h_white["L1"], 50, axis=0),
        np.percentile(h_white["L1"], 95, axis=0),
    ]
).T
np.savetxt(
    "../../Results/Results_for_Fig1/median_combined_waveform_with_90pct_uncertainty.dat", data, delimiter="\t", header="\t".join(
        ["H1_time", "H1_lower", "H1_median", "H1_upper", "L1_time", "L1_lower", "L1_median", "L1_upper"]
    )
)
