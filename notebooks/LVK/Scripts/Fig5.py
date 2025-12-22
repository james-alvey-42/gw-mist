#Script to reproduce Fig.5 

import matplotlib.patches as mpatches
from matplotlib import rcParams, colors
import seaborn as sns
from distutils.spawn import find_executable

import os, h5py, pandas as pd
import fnmatch
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

if find_executable('latex'): rcParams["text.usetex"] = True
rcParams["xtick.direction"] = "inout"
rcParams["ytick.direction"] = "inout"
rcParams["legend.frameon"]  = False
rcParams["legend.loc"]      = "best"
rcParams["axes.grid"]       = True
rcParams["grid.alpha"]      = 0.6
rcParams["grid.linestyle"]  = "dotted"
rcParams["lines.linewidth"] = 0.7
rcParams["axes.titlepad"]   = 30.
rcParams["font.family"]     = "Latin Modern Roman"
rcParams["font.weight"]     = 'bold'

rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18
rcParams["legend.fontsize"] = 18
rcParams["axes.labelsize"]  = 18
rcParams["font.size"]       = 18

parameters      = ['f_t_0', 'tau_t_0']
SampDataFrame   = pd.DataFrame(columns = parameters)
single_evt_keys = {'event': str(), 'pipeline': str(), 'model': str(), 'submodel': str(), 'time': str(), 'GR_tag': str()}
IMR_keys        = {'event': str(), 'pipeline': str(), 'model': str()}

#import posteriors
DICT_PATH = {'GW231123_production_DS_1mode_26M_GR' : '../Results/pyRing_posteriors/posterior_26M.dat',
             'GW231123_production_DS_1mode_28M_GR' : '../Results/pyRing_posteriors/posterior_28M.dat',
             'GW231123_production_DS_1mode_30M_GR' : '../Results/pyRing_posteriors/posterior_30M.dat',
             'GW231123_production_IMR-combined'    : '../Results/posterior_samples.h5'}

def read_posteriors_event(file_path, parameters):
    '''
    Read the posteriors distribution of a single file.
    The posterior distributions for the passed parameters are returned in a Pandas DF.
    '''
    filename = os.path.basename(os.path.normpath(file_path))
    if ('.txt' in filename) or ('.dat' in filename):
        load = np.genfromtxt(file_path, names = True)
    if '.h5' in filename:
        with h5py.File(file_path, 'r') as f:
            try:
                tmp = f['bilby-NRSur7dq4']['posterior_samples']
            except:
                tmp = f['C00:Mixed']['posterior_samples']
            load = np.array(tmp)

    df = pd.DataFrame(load)

    if (set(['final_mass', 'final_spin']) <= set(df.keys())):
        df.rename(columns = {'final_mass': 'Mf', 'final_spin': 'af'}, inplace=True)

    df = compute_qnms_from_remnant(df)
    df = Adapt_Samples(df, parameters)
    df = df.filter(items = parameters)

    df.tau_t_0 *= 1000  # Set time in [ms]

    return df

def compute_qnms_from_remnant(df):
    if not set(['f_t_0', 'tau_t_0']) <= set(df.keys()):
        df = compute_qnms_from_Mf_af(df)
    return df

def Adapt_Samples(df, parameters):
    '''
        Remove unnecessary parameters from the data frame.
    '''
    if not (set(parameters).difference(df.keys()) == set()):
        additional_pars = set(parameters).difference(df.keys())
        for additional_par in additional_pars:
            df.insert(0, additional_par, np.nan)
            df[additional_par][0] = 0

    return df

def downsampling(df):
    '''
    Return the data frame downsampled according to the required probability.
    downsample = 1 takes the 100% of the data, i.e. no downsampling
    '''
    new_nsamp = int(len(df.index) * 0.001)
    df = df.sample(new_nsamp)
    df = df.reset_index()

    return df

def compute_qnms_from_Mf_af(df):
    '''
    Compute QNMs frequency and damping time from Mf and af for one mode (l,m,n)
    using the qnm python package [https://github.com/duetosymmetry/qnm]
    '''
    nsamp = len(df)

    omg = np.zeros(nsamp)
    tau = np.zeros(nsamp)
    for i in range(nsamp):
        Mf, af = df.Mf[i], df.af[i]
        try:
            import pyRing.waveform as wf
        except:
            raise ValueError('Unable to find the pyRing installation for the QNMs fits. Please either install pyRing or disactivate the option "qnms-pyRing".')
        omg[i] = wf.QNM_fit(2, 2, 0).f(Mf, af)                # [Hz]
        tau[i] = wf.QNM_fit(2, 2, 0).tau(Mf, af)              # [ms]

    df.insert(0, 'f_t_0',   omg)
    df.insert(0, 'tau_t_0', tau)

    return df

def hex_to_RGB(hex, alpha):

    RGB = colors.to_rgb(hex)
    RGB += (alpha,)

    return RGB

def labels_legend(par):

    label = ''
    try:
        if   par == '26M':          label = '$38.14 \, \mathrm{[ms]}$'
        elif par == '28M':          label = '$41.08 \, \mathrm{[ms]}$'
        elif par == '30M':          label = '$44.01 \, \mathrm{[ms]}$'
        elif par == 'IMR-combined': label = '$\mathrm{IMR\ combined}$'
        else:
            raise ValueError('Unknown legend parameter.')
    except: label = f'${par}$'

    return label

def corner_plots_sns(SampDataFrame):

    keys = ['26M', '28M', '30M', 'IMR-combined']
    pars = ['f_t_0', 'tau_t_0']
    labels_dict = {'f_t_0': '$f_{1}$ [Hz]', 'tau_t_0': '$\\tau_{1}$ [ms]'}

    number_colors = len(keys)
    palette = ['#56B4E9', '#009E73', '#E69F00', '#000000', '#D55E00']
    colors = sns.color_palette(palette, number_colors)
    SampDataFrame['ordering'] = pd.Categorical(SampDataFrame['time'], categories = keys, ordered = True)

    fig = sns.pairplot(SampDataFrame.sort_values('ordering'),
        corner    = True,
        hue       = 'ordering',
        diag_kind = 'kde',
        vars      = labels_dict,
        palette   = colors,
        height    = 4,
        dropna    = 1,
        plot_kws  = dict(alpha = 0),
        diag_kws  = dict(alpha = 0.5, linewidth = 2, common_norm = False, gridsize= 3000),
    )

    for i, var_x in enumerate(labels_dict):
        for j, var_y in enumerate(labels_dict):
            if j >= i: continue
            ax = fig.axes[i, j]
            for k, key in enumerate(keys):
                sns.kdeplot(
                    data        = SampDataFrame[SampDataFrame['time']==key],
                    x           = var_y,
                    y           = var_x,
                    ax          = ax,
                    levels      = [0.1, 1],
                    fill        = False,
                    color       = colors[k]
                )
                sns.kdeplot(
                    data        = SampDataFrame[SampDataFrame['time']==key],
                    x           = var_y,
                    y           = var_x,
                    ax          = ax,
                    levels      = [0.1, 1],
                    fill        = True,
                    alpha       = 0.5,
                    #linewidth   = 2,
                    color       = colors[k]
                )

    # Add legend
    fig._legend.remove()    # Remove default legend

    patch = [mpatches.Patch(facecolor = colors[ci], edgecolor = 'k', alpha = 0.5, label = labels_legend(c)) for ci,c in enumerate(keys)]
    patch = [mpatches.Patch(facecolor = hex_to_RGB(colors[ci], 0.5), edgecolor = colors[ci], label = labels_legend(c)) for ci,c in enumerate(keys)]
    fig.axes[0, 0].legend(handles = patch, loc = 'center', frameon = False, bbox_to_anchor = (2-0.5, 0.5))

    bounds = [[18,82],[-2,45]]

    for pi,par in enumerate(pars):
        fig.axes[len(pars)-1, pi].set_xlabel(labels_dict[par])
        if not pi==0: fig.axes[pi, 0].set_ylabel(labels_dict[par])
        fig.axes[pi, pi].set_xlim(bounds[pi])
        fig.axes[pi, pi].set_ylim(bounds[pi])

    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass

    path = '../Figures'
    file = 'Fig5_release'
    for extension in ['pdf', 'png']:
        if extension == 'pdf': filename = os.path.join(path, '{name}.{ext}'.format(name = file, ext = extension))
        if extension == 'png': filename = os.path.join(path, '{name}.{ext}'.format(name = file, ext = extension))
        fig.savefig(filename, bbox_inches = 'tight', transparent = True)


for file in tqdm(DICT_PATH.keys(), desc = 'Reading Posteriors'):
    if not ((file == '.DS_Store') or (file == 'noise_evidences') or (file == 'ignore') or (file == 'SNR_samples') or (fnmatch.fnmatch(file, '*IMR*'))):

        file_path = DICT_PATH[file]
        keys = file.split('_')
        for i,key in enumerate(single_evt_keys.keys()):
            single_evt_keys[key] = keys[i]

        EventDataFrame = read_posteriors_event(file_path, parameters)
        EventDataFrame = EventDataFrame.assign(par = single_evt_keys['time'])
        EventDataFrame.rename(columns={'par': 'time'}, inplace = True)
            
        SampDataFrame = pd.concat([SampDataFrame,  EventDataFrame], ignore_index=True)

    # Case when there is one IMR analysis to compare separately.
    if  fnmatch.fnmatch(file, '*IMR*'):

        file_path = DICT_PATH[file]
        keys = file.split('_')
        for i,key in enumerate(IMR_keys.keys()):
            IMR_keys[key] = keys[i]

        EventDataFrame0 = read_posteriors_event(file_path, parameters)
        
        EventDataFrame = EventDataFrame0.assign(par = IMR_keys['model'])
        EventDataFrame.rename(columns={'par': 'time'}, inplace = True)
        SampDataFrame = pd.concat([SampDataFrame, EventDataFrame], ignore_index=True)

corner_plots_sns(SampDataFrame)
