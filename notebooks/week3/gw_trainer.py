### CMDLINE STUFF ###

import argparse
parser = argparse.ArgumentParser(description='Example script')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--name', type=str, help='Path to save model')
parser.add_argument('--correlated', type=bool, default=False, help='Whether the model is trained on correlated distortions')
parser.add_argument('--mac', type=bool, default=False, help='Is this being run on my mac?')
parser.add_argument('--graph_verb', type=bool, default=False, help='Show validation Graphs?')
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

#### SORTING ARGS OUT ####
if args.mac:
    prec = [torch.float32,32]
    dev = 'mps'
    def to_device(nn):
        nn.to(device='mps', dtype=torch.float32).eval()
else:
    prec = [torch.float64,64]
    dev = 'gpu'
    def to_device(nn):
        nn.cuda().eval

print(f'args.mac reads {args.mac}')
print(f'Running at {prec[1]}-bit precision on {dev}')

#### - LOAD THE DATA IN - ###

gen_samples = np.load('generated_samples.npz')

correlation_scales = torch.tensor([5]).int()
gw150814_post = torch.tensor(gen_samples['waveform'])
gw150814_noise = torch.tensor(gen_samples['noise'])

### - CREATE THE SIMULATOR - ###
gw150814_samples = {'mu': gw150814_post, 'noise': gw150814_noise}

if args.correlated:
    simulator = GW150814_Additive(
        gw150814_samples=gw150814_samples, 
        bounds=torch.tensor([1.05]), #1.2341, 0.5696, 0.3403]), 
        dtype=prec[0],
        correlation_scales = correlation_scales
    ) ### For correlated version
else:
    simulator = GW150814_Additive(
        gw150814_samples=gw150814_samples, 
        bounds=2, 
        dtype=prec[0],
        fraction = 0.5
    ) ### For uncorrelated version

print(simulator)

times = simulator.times
Nbins = simulator.Nbins

samples = simulator.sample(500)

#### SANITY CHECK PLOTS #####

if args.graph_verb:
    keys = ['mu','x0','xi',]
    titles = ["Posterior samples","Posterior samples + PSD","Posterior samples + PSD + distortions"]
    for j in range(3):
        fig,ax = pf.create_plot(size=(5,3))
        ax.set_title(titles[j])
        for i in tqdm(range(500)):
            # global x
            # x = times
            # global y
            # y = samples[keys[j]][i].reshape(-1)
            ax.plot(times, samples[keys[j]][i].reshape(-1), color="C0", alpha=0.05)
        ax.plot(simulator.times, simulator.filter_gwosc_data().reshape(-1), color="C1", zorder=10, label="GWOSC data")
        ax.set_xlim(-0.1, 0.1)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Strain")
        ax.legend()
        plt.tight_layout()
        plt.show()

### TRAINING ####

if args.correlated:
    ###### FOR CORRELATED ANALYSIS #######
    ## COMPUTATIONALLY EXPENSIVE (MAC=HOT) ##
    from models.online_norm import OnlineStandardizingLayer
    from models.unet_1d import UNet1d 
    from models.resnet_1d import ResidualNet


    class Network1D(torch.nn.Module):
        def __init__(self, correlation_scales):
            super().__init__()
            
            self.correlation_scales = correlation_scales
            self.online_norm = OnlineStandardizingLayer((len(correlation_scales), Nbins,), use_average_std=True) 
            self.logvariance = torch.nn.Parameter(torch.ones(len(correlation_scales), Nbins)*-2)  # Initialize to big value
                
            self.kernel_sizes = correlation_scales + (correlation_scales % 2 == 0).int()  
            self.paddings = (self.kernel_sizes  - 1)// 2 
            self.net = torch.nn.ModuleList(
                    [ResidualNet(
                    in_features=1, out_features=1, 
                    hidden_features=128, num_blocks=2,
                    dropout_probability=0.2,    
                    kernel_size=self.kernel_sizes[i].item(), padding=self.paddings[i].item()
                    ) for i in range(len(correlation_scales))]
            )
            
        def epsilon(self, x):
            x = self.online_norm(x)
            x = [self.net[i](x[:, i, :].unsqueeze(1)).squeeze(1) for i in range(len(self.correlation_scales))
            ]
            x = torch.stack(x, 1) 
            return x
        
        def snr(self, x):
            return self.epsilon(x) / self.logvariance.exp().sqrt()  # [B, N_bins]
        
        def bounds(self):
            cut = 5
            return self.logvariance.detach().exp().sqrt()[:, cut:-cut].mean(-1) * 5
            
        def forward(self, x):
            data = x['x']
            epsilon_sim = x['epsilon']
            epsilon = self.epsilon(data) #[B, #correlation_scales, N_bins]        
            mask = ( x['ni'] != 0 )  
            squared_error = (epsilon - epsilon_sim)**2                                                                    # [B, N_bins]
            l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance               # [B, N_bins]
            
            return (l * mask.unsqueeze(1).float()).sum() * 0.5
        
    network = Network1D(correlation_scales)
    network.bounds()
    to_device(network)

    #### TRAIN ####
    def resample(sample):
        sample = simulator._resample(sample)
        sample['x'] = sample['xi']
        sample = {k: v for k, v in sample.items()}
        return sample

    batch_size = 128
    dm = OnTheFlyDataModule(simulator, Nsims_per_epoch=500*batch_size, batch_size=batch_size)
    network = Network1D(correlation_scales)
    model = CustomLossModule_withBounds(network)
    trainer = pl.Trainer(
        accelerator=dev, 
        max_epochs=args.epochs, 
        precision=prec[1],
        # fast_dev_run=True
    )
    trainer.fit(model, dm)
    torch.save(model, 'out/'+args.name+'_uc_model')
    torch.save(network, 'out/'+args.name+'_uc_network')
    to_device(network)

else:
    ##### FOR UNCORRELATED ADDITIONS #####
    from models.online_norm import OnlineStandardizingLayer
    from models.resnet_1d import ResidualNet


    class Network_SNR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.online_norm = OnlineStandardizingLayer((Nbins,), use_average_std=True) 
            self.net = ResidualNet(1, 1, 128)
            self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
                    
        def epsilon(self, x):
            x = self.online_norm(x)
            x = self.net(x.unsqueeze(1)).squeeze(1)
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

            epsilon = self.epsilon(data)
            mask = ( x['ni'] != 0 )  
            squared_error = (epsilon - epsilon_sim)**2                                                                    # [B, N_bins]
            l = squared_error / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
            return (l * mask.float()).sum() * 0.5

        # Train
    def resample(sample):
        sample = simulator._resample(sample)
        sample['x'] = sample['xi']
        sample = {k: v for k, v in sample.items()}
        return sample

    batch_size = 128
    # dm = StoredDataModule(samples, batch_size=batch_size, on_after_load_sample=resample)
    dm = OnTheFlyDataModule(simulator, Nsims_per_epoch=400*batch_size, batch_size=batch_size)
    network_SNR = Network_SNR()
    model_SNR = CustomLossModule_withBounds(network_SNR, learning_rate=8e-3)
    trainer = pl.Trainer(
        accelerator=dev, 
        max_epochs=args.epochs, 
        precision=prec[1],
        # fast_dev_run=True
    )
    trainer.fit(model_SNR, dm)
    torch.save(model_SNR, args.name+'uc_model')
    torch.save(network_SNR, args.name+'uc_network')
    to_device(network_SNR)

## test ###