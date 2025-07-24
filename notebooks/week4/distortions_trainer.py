import argparse
parser = argparse.ArgumentParser(description='Script to train NN on GW Data')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--name', type=str, help='Path to save model')
parser.add_argument('--mac', action='store_true', help='Is this being run on my mac?')
parser.add_argument('--psd', action='store_true', help='Is this being run on my mac?')
args = parser.parse_args()

import sys
sys.path.append('../../mist-base/GW')
sys.path.append('../../mist-base/')
sys.path.append('../../mist-base/utils')
sys.path.append('../../')

import gw150814_simulator as gs
from gw150814_simulator import GW150814, defaults, GW150814_Additive
# import module

import torch
torch.set_float32_matmul_precision('medium')
import numpy as np
import multiprocessing as mp
# mp.set_start_method("spawn", force=True)
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

#### SORTING ARGS OUT ####
if args.mac:
    prec = [torch.float32,32]
    dev = 'mps'
    wkrs = 0
    def to_device(nn):
        nn.to(device='mps', dtype=torch.float32).eval()
else:
    prec = [torch.float64,64]
    dev = 'gpu'
    wkrs = 31
    def to_device(nn):
        nn.cuda().eval

# corr_marker = 'correlated' if args.correlated else 'uncorrelated'
title_marker = f'BLANK' if args.name==None else args.name
psd_marker = f'PSD' if args.psd==True else f'white'


print(f'Running a '+psd_marker+f' simulation of {args.epochs} epochs.')
print(f'you are running at {prec[1]}-bit precision on {dev}. Title:'+title_marker)

import torch
import numpy as np
import scipy
import pytorch_lightning as pl
from tqdm import tqdm

import matplotlib.pyplot as plt

import os, sys
sys.path.append('../../mist-base/GW')
sys.path.append('../../mist-base/')
sys.path.append('../../mist-base/utils')
sys.path.append('../../')

from src.utils.generators import Simulator_Additive
from simulators.utils import *
from utils.data import OnTheFlyDataModule, StoredDataModule
from utils.module import CustomLossModule_withBounds, BCELossModule

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

mycolors = ['#570f6d', '#bb3754', '#f98e08']

modes = ['gw','white','complex']
for mode in modes:
    Nsims = 100_000
    Nbins = 100
    SIGMA = 1 # Likelihood covariance \Sigma = sigma * \mathbb{1}
    simulator = Simulator_Additive(Nbins=Nbins, sigma=SIGMA, bounds=5, fraction=0.2, bkg=True, dtype=torch.float64, mode=mode)     
    Nbins = simulator.Nbins
    samples = simulator.sample(Nsims=Nsims)  

    from models.online_norm import OnlineStandardizingLayer
    from models.resnet_1d import ResidualNet

    class Network_epsilon(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
            self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        def forward(self, x):
            data = x['x']
            x = self.net(data.unsqueeze(1)).squeeze(1)
            return x
                    
        def epsilon(self, x):
            x = self.net(x.unsqueeze(1)).squeeze(1) # x-net
            return x
        
        def snr(self, x):
            return self.epsilon(x) / self.logvariance.exp().sqrt()  # [B, N_bins]
        
        def bounds(self):
            return self.logvariance.detach().exp().sqrt().mean(-1) * 5
            
        def forward(self, x):
            
            # Adaptive data generation
            ni = x['ni']
            epsilon_sim =  (2 * self.bounds() * torch.rand(x['x'].shape, device= x['x'].device, dtype= x['x'].dtype) - self.bounds()) * ni
            data =  x['x0'] + epsilon_sim * ni
            
            # data = x['x']
            epsilon = self.epsilon(data)
            mask = ( x['ni'] != 0 )  
            squared_error = (epsilon - epsilon_sim)**2                                                  # [B, N_bins]
            l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
            return (l * mask.float()).sum() * 0.5
        
    batch_size = 128
    samples = simulator.sample(Nsims=Nsims)  
    # dm = StoredDataModule(samples, batch_size=batch_size, on_after_load_sample=resample)
    dm = OnTheFlyDataModule(simulator, Nsims_per_epoch=400*batch_size, batch_size=batch_size, num_workers=31)
    network_epsilon = Network_epsilon()
    model = CustomLossModule_withBounds(network_epsilon, learning_rate=3e-3)
    trainer = pl.Trainer(
        accelerator="gpu", 
        max_epochs=20, 
        precision=64,
        # fast_dev_run=True
    )
    trainer.fit(model, dm)
    network_epsilon.cuda().eval();

    torch.save(model, 'out/'+title_marker+'_'+mode+'_uc_model')
    torch.save(network_epsilon, 'out/'+title_marker+'_'+mode+'_uc_network')