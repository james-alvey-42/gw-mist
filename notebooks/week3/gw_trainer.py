### CMDLINE STUFF ###

import argparse
parser = argparse.ArgumentParser(description='Example script')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--name', type=str, help='Path to save model')
parser.add_argument('--additive', type=bool, default=False, help='Whether the model is trained on additive distortions')
args = parser.parse_args()

import sys
sys.path.append('../../mist-base/GW')
sys.path.append('../../mist-base/')
sys.path.append('../../mist-base/utils')

import gw150814_simulator as gs
from gw150814_simulator import GW150814, defaults, GW150814_Additive
# import module

import torch
import numpy as np
import scipy
import scipy.stats
import pytorch_lightning as pl
from collections import defaultdict
from tqdm import tqdm
import jax.numpy as jnp
import plotfancy as pf
pf.housestyle_rcparams()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os, sys

from simulators.additive import Simulator_Additive
from simulators.utils import *
from utils.data import OnTheFlyDataModule, StoredDataModule
from utils.module import CustomLossModule_withBounds, BCELossModule

mycolors = ['#77aca2', '#ff004f', '#f98e08']
#### - LOAD THE DATA IN - ###

correlation_scales = torch.tensor([5]).int()
gw150814_post = torch.tensor(tdw)
gw150814_noise = torch.tensor(tdn)

### - CREATE THE SIMULATOR - ###

gw150814_samples = {'mu': gw150814_post, 'noise': gw150814_noise}
# simulator = GW150814_Additive(
#     gw150814_samples=gw150814_samples, 
#     bounds=torch.tensor([1.05]), #1.2341, 0.5696, 0.3403]), 
#     dtype=torch.float32,
#     correlation_scales = correlation_scales
# ) ### For correlated version
simulator = GW150814_Additive(
    gw150814_samples=gw150814_samples, 
    bounds=2, 
    dtype=torch.float32,
    fraction = 0.5
) ### For uncorrelated version

times = simulator.times
Nbins = simulator.Nbins

