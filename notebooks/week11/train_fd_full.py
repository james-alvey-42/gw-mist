import torch
import numpy as np
import scipy
import pytorch_lightning as pl
from tqdm import tqdm
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
torch.set_float32_matmul_precision('medium')
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import multiprocessing
# multiprocessing.set_start_method('spawn')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os, sys

sys.path.append("../../mist-base")
sys.path.append('../../')

from src.simulators.fd_simulators import GW_Additive_F, GW_Additive_F_Correlated
from simulators.utils import *
from utils.data import OnTheFlyDataModule, StoredDataModule
from utils.module import CustomLossModule_withBounds, BCELossModule, NewLossModule_withBounds
from models.online_norm import OnlineStandardizingLayer
from models.resnet_1d import ResidualNet

import plotfancy as pf

mycolors = ['#570f6d', '#bb3754', '#f98e08']
quantiles = np.array([0.05199686, 0.2145543,  0.58780088, 1.17737921, 1.91882754,
 2.75067576, 3.63463655])

quantiles_long = np.array([7.11978022e-04, 7.96148769e-03, 5.19968566e-02, 2.14554300e-01,
 5.87800876e-01, 1.17737921e+00, 1.91882754e+00, 2.75067576e+00,
 3.63463655e+00, 4.55164698e+00, 5.49045819e+00])
# simulator_plot = GW_Additive_F(bkg=True, fraction=0.002)
simulator = GW_Additive_F(bkg=True, fraction=0.2)



# test = simulator_plot.sample(1)

# pf.housestyle_rcparams()
# fig, ax1 = pf.create_plot()

# plt.setp(ax1.get_xticklabels(), visible=False)
# ax2 = fig.add_axes((0,-.3,1,0.3), sharex=ax1)
# ax3 = fig.add_axes((1,-.3,0.2,0.3), sharey=ax2)
# plt.setp(ax3.get_xticklabels(), visible=False)
# plt.setp(ax3.get_yticklabels(), visible=False)
# ax4 = fig.add_axes((0,1,1,0.3), sharex=ax1)
# plt.setp(ax4.get_xticklabels(), visible=False)


# ax1.plot(simulator.grid,torch.abs(test['x0'][0]), label=r'$x_0$', color='#ff004f', lw=0.5)
# ax1.plot(simulator.grid,torch.abs(test['mu'][0]), label=r'$\mu$', color='black')
# ax1.set_ylabel(r'$|\tilde{d}(f)|$')
# ax1.set_ylim([-.2,8])
# ax1.legend(loc='upper right')
# ax1.set_xlim(20, 800)


# resd = torch.abs(test['x0'][0]-test['mu'][0])
# ax2.plot(simulator.grid,resd, color='#ff004f', lw=0.5)
# ax2.set_xlabel(r'$f$ [Hz]')
# ax2.set_ylabel(r'res ($x_0$)')
# ax2.set_ylim([0,2.5])
# for i in range(1,6):
#     ax1.fill_between(simulator.grid, quantiles_long[i]+torch.abs(test['mu'][0]), quantiles_long[-i]+torch.abs(test['mu'][0]),  color='#b0b0b0', alpha=0.15)
#     ax2.fill_between(simulator.grid, quantiles_long[i], quantiles_long[-i],  color='#b0b0b0', alpha=0.15)
#     ax3.fill_between(simulator.grid, quantiles_long[i], quantiles_long[-i],  color='#b0b0b0', alpha=0.15)

# ax3.hist(resd, orientation='horizontal', bins=torch.linspace(0,3, 15), edgecolor='black', color='#ff004f', density=True)
# ax3.set_xlim([0,1])

# for i in range(100):
#     ax4.plot(simulator.grid,torch.abs(simulator.sample(1)['mu'][0]), lw=0.5, color='black', alpha=0.05)


# ax4.set_yticks([])
# ax4.set_ylim([0,2])
# ax4.set_visible(False if simulator.bkg==False else True)
# pf.fix_plot([ax1,ax2, ax3,ax4])

# ax1.plot(simulator.grid,torch.abs(test['xi'][0]), label=r'$x_i$', color="#d931f3", lw=0.5, zorder=-1)

class Network_epsilon_complex(torch.nn.Module):
    def __init__(self, nbins):
        super().__init__()
        
        self.nbins = nbins
        self.logvariance = torch.nn.Parameter(torch.ones(self.nbins, dtype=torch.float64) * 5)
        
        self.net = ResidualNet(2, 2, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) #now takes 2 input channels (real, imag) and outputs 1 (real epsilon)
        self.mu_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.nbins * 2, 128), # Input is concatenated real and imaginary parts
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.nbins * 2)  # Output is concatenated real and imaginary parts
        )

    def mu(self, x):
        x_real_imag = torch.cat([x.real, x.imag], dim=-1) # Shape: [B, 2 * nbins]
        out_real_imag = self.mu_predictor(x_real_imag) # Shape: [B, 2 * nbins]
        pred_real, pred_imag = torch.chunk(out_real_imag, 2, dim=-1) # Each is [B, nbins]
        return torch.complex(pred_real, pred_imag)
 
    
    def epsilon(self, x):
        resd = x - self.mu(x)
        resd_channels = torch.stack([resd.real, resd.imag], dim=1)
        out_channels = self.net(resd_channels)
        return torch.complex(out_channels[:, 0, :], out_channels[:, 1, :])
    
    def snr(self, x):
        return torch.abs(self.epsilon(x)) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        x0_block = torch.nan_to_num(x['x0'], nan=0, posinf=0, neginf=0) # Expected to be complex
        mu_block = torch.nan_to_num(x['mu'], nan=0, posinf=0, neginf=0) # Expected to be complex
        ni = torch.nan_to_num(x['ni'], nan=0, posinf=0, neginf=0)
        
        rand_real = 2 * torch.rand(x0_block.shape, device=x0_block.device) - 1
        rand_imag = 2 * torch.rand(x0_block.shape, device=x0_block.device) - 1
        epsilon_sim_unit = torch.complex(rand_real, rand_imag)
        epsilon_sim = self.bounds()*epsilon_sim_unit*ni

        data = x0_block + epsilon_sim # data is complex
        
        # net evaluation_m
        net_mu = self.mu(data) # net_mu is complex
        
        # Loss for complex numbers is the squared magnitude of the difference
        error_mu = (torch.abs(net_mu - mu_block))**2
        
        l_mu = error_mu / (self.logvariance.exp() + 1e-10) + self.logvariance
        l_mu_return = l_mu.sum() * 0.5

        # net evaluation_e
        net_epsilon = self.epsilon(data) # net_epsilon is real
        mask = ( ni != 0 )
        
        # epsilon_sim is real, net_epsilon is real. Standard squared error.
        squared_error_e = (torch.abs(net_epsilon - epsilon_sim))**2                                     # [B, N_bins]
        l_e = squared_error_e / (self.logvariance.exp() + 1e-21) + self.logvariance               # [B, N_bins]
        l_e_return = (l_e * mask.float()).sum() * 0.5
        
        # combine
        return l_mu_return + l_e_return
batch_size = 128

dm = OnTheFlyDataModule(simulator, Nsims_per_epoch=400*batch_size, batch_size=batch_size, num_workers=31)

network_epsilon = Network_epsilon_complex(nbins=simulator.Nbins)
model = NewLossModule_withBounds(network_epsilon, learning_rate=3e-3)
trainer = pl.Trainer(
    accelerator="gpu", 
    max_epochs=25, 
    precision=64,
    # fast_dev_run=True
)
trainer.fit(model, dm)
network_epsilon.cuda().eval();


############################################################################################################
############################################################################################################
############################################################################################################

torch.save(network_epsilon, f'networks/network_GW_complex')
torch.save(model, f'networks/model_GW_complex')
netid = 'GW_complex'

############################################################################################################
############################################################################################################
############################################################################################################

# Convert tensors to scalars if they are tensors
train_loss_history = [loss.item() if hasattr(loss, 'item') else loss for loss in model.train_loss_history]
bounds_history = [bound.item() if hasattr(bound, 'item') else bound for bound in model.bounds_history]

# Generate a list of epoch numbers
epochs = range(1, len(train_loss_history) + 1)

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
# Plot Training Loss over Epochs
axs[0].plot(epochs, train_loss_history)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Training Loss')
# Plot Bounds over Epochs
axs[1].plot(epochs, bounds_history, label='Bounds', color='orange')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Bounds')


plt.tight_layout()
plt.savefig(f'figs/{netid}/bounds.png', dpi=300)
plt.tight_layout();

# Distotions enveloping the data
bounds_history = [bound.item() if hasattr(bound, 'item') else bound for bound in model.bounds_history]

fig, axs = plt.subplots(4, 5, figsize=(20, (4+1)*3), sharex=True)
axs = axs.flatten()
for i_b, b in enumerate(bounds_history):
    if i_b < len(axs):
        axs[i_b].set_title(r"$b$ = {:.2f}".format(b))
        for j in range(10):
            sample = simulator.sample(1)
            x0_block = torch.nan_to_num(x['x0'], nan=0, posinf=0, neginf=0) # Expected to be complex
            mu_block = torch.nan_to_num(x['mu'], nan=0, posinf=0, neginf=0) # Expected to be complex
            ni = torch.nan_to_num(x['ni'], nan=0, posinf=0, neginf=0)
            
            rand_real = 2 * torch.rand(x0_block.shape, device=x0_block.device) - 1
            rand_imag = 2 * torch.rand(x0_block.shape, device=x0_block.device) - 1
            epsilon_sim_unit = torch.complex(rand_real, rand_imag)
            epsilon_sim = b*epsilon_sim_unit*ni

            data = x0_block + epsilon_sim # data is complex
            axs[i_b].plot(data[0].cpu(), c='C0', alpha=0.4)

plt.tight_layout()
plt.savefig(f'figs/{netid}/history.png', dpi=300)