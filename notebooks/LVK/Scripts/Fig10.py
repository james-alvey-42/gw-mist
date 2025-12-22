#Script to reproduce Fig 10
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy, pylab
import pandas
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
import h5py, bilby, os
from gwpy.timeseries import TimeSeries
from pesummary.io import read
from rich.progress import track
from matplotlib.ticker import ScalarFormatter
from joblib import Parallel, delayed
from astropy.constants import G, M_sun, c
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set_context('talk') 
sns.set_theme(font_scale=1.2)
sns.set_palette('colorblind')
sns.set_style('ticks')

colors = ["#3690c8","#f8db37","#ebab78","#e4607b","#f2efed","#8fb5c7","#352c30","#c0d3e0","#eddab1","#d99fad"]

pylab.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'axes.grid' : True,
        'grid.linestyle' : ':',
        'grid.color' : '#bbbbbb'

    }
)

pylab.rcParams['axes.linewidth'] = 1
pylab.rcParams['axes.prop_cycle'] = pylab.cycler(color=colors)


#Load posterior samples
result_file = "../Results/posterior_samples.h5"
read_result = read(result_file)
samples = read_result.samples_dict

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
    _data = TimeSeries.get(channel=channel_dict[ifo.name],start=1384782882.634277, end=1384782890.634277, verbose=True, allow_tape=True)
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

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration = 8.0, 
    sampling_frequency = 1024.0,
    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
    start_time=1384782888.634277 - 6.,
    waveform_arguments={
        "reference_frequency": 10.0,
        "minimum_frequency": 10.0,
        "maximum_frequency": 448.0,
        "waveform_approximant": "NRSur7dq4",
    },)



posteriors = pandas.DataFrame.from_dict(samples["C00:NRSur7dq4"])

detector = {'H1' : 'Hanford', 'L1' : 'Livingston'}

def get_posterior_waveform(
    ifo,
    number_of_samples,
    waveform_generator,
    posterior_samples,
    peak_time = None,
    start_time=-0.15,
    end_time = 0.1,
    additionl_time=None,
    axes=None,):

    if axes is None:
        axes = pylab.gca()
    
    maxl = numpy.argmax(posterior_samples["log_likelihood"])
    geocent_time = posterior_samples["geocent_time"][maxl]

    time_domain_waveforms = []
    for k in track(range(number_of_samples), description=f"Generating {ifo.name} waveforms"):
        parameters = posterior_samples.iloc[k].to_dict()
        waveform_polarisations = waveform_generator.frequency_domain_strain(parameters)
        frequency_domain_waveform = ifo.get_detector_response(waveform_polarisations, parameters)
        time_domain_waveform = bilby.core.utils.infft(
            frequency_domain_waveform,
            sampling_frequency=ifo.sampling_frequency,
        )
        time_domain_waveforms.append(time_domain_waveform)
    time_domain_waveforms = numpy.array(time_domain_waveforms)

    time_shifted = ifo.time_array - geocent_time
    mean_waveform = numpy.mean(time_domain_waveforms, axis=0)
    lower_bound = numpy.percentile(time_domain_waveforms, 5, axis=0)
    upper_bound = numpy.percentile(time_domain_waveforms, 95, axis=0)
    axes.plot(
        time_shifted,
        mean_waveform,
        label='Mean posterior waveform')
    axes.fill_between(
        time_shifted,
        lower_bound,
        upper_bound,
        alpha=0.3,
        label='90% Credible Interval')
    
    # Strain morphology peak
    samples_peaks = time_shifted[numpy.argmax(numpy.abs(time_domain_waveforms), axis=1)]
    morphology_peak_median = numpy.median(samples_peaks)
    morphology_peak_lower = numpy.percentile(samples_peaks, 5)
    morphology_peak_upper = numpy.percentile(samples_peaks, 95)
    axes.axvline(
        morphology_peak_median,
        color='k', linestyle='-.', alpha=0.7)
    axes.axvspan(
        morphology_peak_lower,
        morphology_peak_upper,
        color='k', alpha=0.1, label=r'$t_\mathrm{peak}^\mathrm{strain}$ (90%)')
    
    # Polarisation peak
    t_peak_relative = peak_time - geocent_time
    axes.axvline(t_peak_relative,
                 color='#ca2000', linestyle='--')
    axes.axvspan(t_peak_relative - 0.0153, t_peak_relative + 0.0111, color='#ca2000', alpha=0.3, label=r'$t_\mathrm{peak}^\mathrm{pol}$ (90%)')

    # Formatting
    axes.set_xlim(start_time, end_time)
    axes.set_title(f'{detector[ifo.name]}', fontsize=20)
    axes.set_ylabel(r'$h(t)$', fontsize=20)
    axes.set_xlabel(r'Time [s]', fontsize=20)
    axes.legend(loc='lower center', ncol=5, fontsize=20,  bbox_to_anchor=(0.5, -0.5))

    # Use scalar formatter with math text for scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    axes.yaxis.set_major_formatter(formatter)
    axes.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    sns.despine()
    pylab.tight_layout()


t_peak = {'H1' : 1384782888.59979, # Greg provided these
          'L1' : 1384782888.59579}
fig, ax = pylab.subplots(figsize=(15, 6))
get_posterior_waveform(ifo=ifos[0], 
                       number_of_samples=5000, 
                       waveform_generator=waveform_generator,
                       posterior_samples=posteriors,
                       peak_time=t_peak[ifos[0].name],
                       start_time=-0.15, end_time=0.1)

fig.savefig('../Figures/Fig10_top.pdf', bbox_inches='tight')

#### -----------------

def get_posterior_waveform_modes(mode_array, 
                                 posterior_samples,
                                 ifo,
                                 number_of_samples,):
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration = 8.0, 
        sampling_frequency = 1024.0,
        frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
        start_time=1384782888.634277 - 6.,
        waveform_arguments={
            "reference_frequency": 10.0,
            "minimum_frequency": 10.0,
            "maximum_frequency": 448.0,
            "waveform_approximant": "NRSur7dq4",
            'mode_array' : mode_array,
        },)

    frequency_domain_strain, time_domain_strains = [], []
    whitened_frequency_domain_strain, whitened_time_domain_strain = [], []
    time_array = ifo.time_array
    frequency_array = ifo.frequency_array
    for k in track(range(number_of_samples), description=f"Generating {ifo.name} waveforms"):
        parameters = posterior_samples.iloc[k].to_dict()
        waveform_polarisations = waveform_generator.frequency_domain_strain(parameters)
        frequency_domain_waveform = ifo.get_detector_response(waveform_polarisations, parameters)
        whitened_frequency_domain_waveform = frequency_domain_waveform / (ifo.amplitude_spectral_density_array * numpy.sqrt(ifo.duration / 4))
        whitened_frequency_domain_strain.append(whitened_frequency_domain_waveform)
        frequency_domain_strain.append(frequency_domain_waveform)
        time_domain_waveform = bilby.core.utils.infft(
            frequency_domain_waveform,
            sampling_frequency=ifo.sampling_frequency,
        )
        whitened_time_domain_waveform = bilby.core.utils.infft(
            whitened_frequency_domain_waveform,
            sampling_frequency=ifo.sampling_frequency,
        )
        whitened_time_domain_strain.append(whitened_time_domain_waveform)
        time_domain_strains.append(time_domain_waveform)
    time_domain_strains = numpy.array(time_domain_strains)
    frequency_domain_strain = numpy.array(frequency_domain_strain)

    whitened_frequency_domain_strain = numpy.array(whitened_frequency_domain_strain)
    whitened_time_domain_strain = numpy.array(whitened_time_domain_strain)

    return (str(mode_array), {
        "frequency_domain_strain": frequency_domain_strain,
        "time_domain_strain": time_domain_strains,
        'whitened_time_domain_strain' : whitened_time_domain_strain,
        'whitened_frequency_domain_strain' : whitened_frequency_domain_strain,
        "time_array": time_array,
        "frequency_array": frequency_array,
    })


def get_mode_by_mode_parallel(ifo, posteriors, num_jobs=10, number_of_samples=100):
    mode_arrays = [[(2,2), (2,-2)], [(3,3), (3,-3)], [(2,1), (2,-1)], [(4,4), (4,-4)], [(2,0,)]]
    bilby.core.utils.setup_logger(log_level='ERROR')
    results_list = Parallel(n_jobs=num_jobs)(
        delayed(get_posterior_waveform_modes)(mode_array, 
                                 posteriors,
                                 ifo,
                                 number_of_samples,)
        for mode_array in mode_arrays
    )
    return dict(results_list)

mode_strain_data = get_mode_by_mode_parallel(ifos[0], number_of_samples=5000, posteriors=posteriors)

target_modes = ['[(4, 4), (4, -4)]', '[(3, 3), (3, -3)]', '[(2, 0)]']
# --- Config ---
fontsize_main = 20
fontsize_tick = 16
fontsize_legend = 18
figsize = (16, 8)

maxl = numpy.argmax(posteriors["log_likelihood"])
geocent_time = posteriors["geocent_time"][maxl]

lower_percentile = 5
upper_percentile = 95

difference_between_t_peak_and_mode_by_mode_definition = 8.094536942987238 
constant = G.value * M_sun.value / c.value**3
difference_in_seconds = constant * posteriors['total_mass'][maxl] * difference_between_t_peak_and_mode_by_mode_definition

#to reproduce the orders (and colors) of the paper
mode_keys = ['[(2, 0)]', '[(2, 1), (2, -1)]', '[(2, 2), (2, -2)]', '[(3, 3), (3, -3)]', '[(4, 4), (4, -4)]']

spectral_colors = sns.color_palette("colorblind", n_colors=len(mode_strain_data.keys()))
mode_color_map = {mode: spectral_colors[i] for i, mode in enumerate(mode_keys)}


fig = pylab.figure(figsize=figsize)
outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)

ax_strain = fig.add_subplot(outer_gs[0])
ax_white = fig.add_subplot(outer_gs[1], sharex=ax_strain)

# Inset axes
inset_strain = inset_axes(ax_strain, width="60%", height="60%",
                            bbox_to_anchor=(1.05, 0.55, 0.4, 0.4),
                            bbox_transform=fig.transFigure,
                            loc='upper left', borderpad=0)
inset_white = inset_axes(ax_white, width="60%", height="60%",
                            bbox_to_anchor=(1.05, 0.15, 0.4, 0.4),
                            bbox_transform=fig.transFigure,
                            loc='upper left', borderpad=0)

for mode_key in mode_keys:
    mode_data = mode_strain_data[mode_key]
    time_array = mode_data["time_array"][()]
    time_strain = numpy.array(mode_data["time_domain_strain"])
    whitened_strain = numpy.array(mode_data["whitened_time_domain_strain"])
    shifted_time = time_array - geocent_time

    lower_time = numpy.percentile(time_strain, lower_percentile, axis=0)
    upper_time = numpy.percentile(time_strain, upper_percentile, axis=0)
    lower_white = numpy.percentile(whitened_strain, lower_percentile, axis=0)
    upper_white = numpy.percentile(whitened_strain, upper_percentile, axis=0)

    ax_strain.fill_between(shifted_time, lower_time, upper_time, alpha=0.4,
                            color=mode_color_map[mode_key], label=mode_key)
    ax_white.fill_between(shifted_time, lower_white, upper_white, alpha=0.4,
                            color=mode_color_map[mode_key])

    if mode_key in target_modes:
        inset_strain.fill_between(shifted_time, lower_time, upper_time, alpha=0.4,
                                    color=mode_color_map[mode_key], label=mode_key)
        inset_white.fill_between(shifted_time, lower_white, upper_white, alpha=0.4,
                                    color=mode_color_map[mode_key])

# Peak markers
for ax in [ax_strain, ax_white, inset_strain, inset_white]:
    ax.axvline(t_peak['H1'] - geocent_time, color='#ca2000', linestyle='-', linewidth=1)
    ax.axvline(t_peak['H1'] - geocent_time + difference_in_seconds, color='#171717', linestyle='-.', linewidth=1)
    ax.set_xlim(-0.05, 0.1)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=fontsize_tick)

# Axis labels
ax_strain.set_ylabel(r"$h(t)$", fontsize=fontsize_main)
ax_white.set_ylabel(r"$\sigma_\mathrm{noise}$", fontsize=fontsize_main)
ax_white.set_xlabel(r"Time [s]", fontsize=fontsize_main)

# Insets
inset_strain.set_xlim(-0.05, 0.1)
inset_white.set_xlim(-0.05, 0.1)
inset_strain.tick_params(axis='both', labelsize=fontsize_tick)
inset_white.tick_params(axis='both', labelsize=fontsize_tick)

ax_strain.indicate_inset_zoom(inset_strain, edgecolor="#7C6E7F")
ax_white.indicate_inset_zoom(inset_white, edgecolor="#7C6E7F")

# Formatter: use LaTeX-style 10^-22 format
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
for ax in [ax_strain, ax_white, inset_strain, inset_white]:
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

handles, labels = ax_strain.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=fontsize_legend, bbox_to_anchor=(0.5, -0.11))
sns.despine()
pylab.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('../Figures/Fig10_bottom.pdf', bbox_inches='tight')

