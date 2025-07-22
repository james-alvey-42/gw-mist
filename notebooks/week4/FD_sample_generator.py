import argparse
parser = argparse.ArgumentParser(description='FD Generator for the GW Trainer Script')
parser.add_argument('--nsamples', type=int, default=1000, help='Number of samples gnerated')
args = parser.parse_args()

import sys
sys.path.append('../../mist-base/GW')
sys.path.append('../../mist-base/')
sys.path.append('../../mist-base/utils')

import gw150814_simulator as gs

import numpy as np
from tqdm import tqdm
import plotfancy as pf
pf.housestyle_rcparams()

import sys

default = gs.defaults
default['posterior_samples_path'] = '../../mist-base/GW/GW150814_posterior_samples.npz'
default['f_max']=250
gw = gs.GW150814(settings=default)
l = len(gw.time_to_frequency_domain(gw.generate_time_domain_waveform()))
s = args.nsamples
fdw = np.zeros([s,l])
fdn = np.zeros([s,l])

for i in tqdm(range(s)):
    fdw[i] = gw.time_to_frequency_domain(gw.generate_time_domain_waveform())
    fdn[i] = gw.time_to_frequency_domain(gw.generate_time_domain_noise())

np.savez('FD_samples.npz', fdw='waveform',fdn='noise')