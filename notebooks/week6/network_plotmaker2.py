######## PRELIMS #########

import torch
torch.set_float32_matmul_precision('medium')
import numpy as np
import scipy
from scipy.stats import norm
from scipy.interpolate import interp1d
import plotfancy as pf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from cycler import cycler
import argparse
pf.housestyle_rcparams()

import os, sys
sys.path.append('../../mist-base/GW')
sys.path.append('../../mist-base/')
sys.path.append('../../mist-base/utils')
sys.path.append('../../')

from src.utils.generators import Simulator_Additive
from simulators.utils import *
from utils.module import CustomLossModule_withBounds, BCELossModule

pf.housestyle_rcparams()

mycolors = ['#570f6d', "#9e8f92", '#f98e08']
folly = '#ff004f'
mygold = (218/255, 165/255, 64/255, 0.1)  

############ FUNCTIONs #########

def cax(ax, pad=0.05):
    divider = make_axes_locatable(ax)
    return divider.append_axes(position='top', size='5%', pad=pad)

def t_to_pvalue(ts, df=1):
    return 1 - scipy.stats.chi2.cdf(ts, df)

def t_to_pvalue_empirical(ts_obs, ts0_ref):
    ts0_ref = np.asarray(ts0_ref)
    return (np.sum(ts0_ref >= ts_obs) + 1) / (len(ts0_ref) + 1)

def dict_to_mps(x):
    if isinstance(x, dict):
        return {k: dict_to_mps(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device='mps')
    elif isinstance(x, list):
        return [dict_to_mps(item) for item in x]
    else:
        return x

def dict_to_cpu(x: dict) -> dict:
    out = {}
    for key, val in x.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device='cpu')
        else:
            out[key] = val
    return out


def ts_sbi(x: dict, model: int =0):
    # x = dict_to_double(x)
    x = dict_to_mps(x)
    
    # Test statistic for sample 0 or 1
    if 'x' not in x.keys():
        if model == 0:
            x['x'] = x['x0']
        elif model == 1:
            x['x'] = x['xi'] 

    # Test statistic
    t = 2 * (network_BCE(x).detach().cpu().squeeze(0).numpy())
    
    # Go back to cpu 
    x = dict_to_cpu(x)
    return t

def chop_middle(array, remove=5, linemode=True):
    if len(array)%2==0:
        mid_u = int(len(array)/2)
        mid_d = mid_u -1
        if not linemode:
            return np.concatenate([array[:mid_d-remove], array[mid_u+remove:]])
        else:
            return array[:mid_d-remove] , array[mid_u+remove:], array[mid_u]
    else:
        mid = len(array)//2
        if not linemode:
            return np.concatenate([array[:mid-remove], array[mid+remove:]])
        else:
            return array[:mid-remove], array[mid+remove:], array[mid]
        
def get_snr2(input:dict):
    target = input['x']
    snr2_nn = network_epsilon.snr(target.to(device='mps', dtype=torch.float32)).detach().cpu().numpy()**2 
    return snr2_nn

def local_do_ticks(list_of_axes, dir = 'in'):
    for ax in list_of_axes:
        ax.minorticks_on()
        ax.tick_params(top=True,right=True, direction=dir, length=7, which='major')
        ax.tick_params(top=True,right=True, direction=dir, length=4, which='minor')

def local_fix_frame(ax):
    ax.tick_params(color='black', labelcolor='black')
    ax.spines[:].set_color('black')
    ax.spines[:].set_linewidth(1)
    return True

def local_fix_plot(a, tickdir ='in'):
    for axes in a:
        axes.grid(False)
        local_do_ticks([axes], tickdir)
        local_fix_frame(axes)
    return True


######## GLOBALS #########

parser = argparse.ArgumentParser(description='Script to make plots post NN training')
# numbers
parser.add_argument('--sigma', type=float, default=1, help='White Noise Sigma (unused if complex)')
parser.add_argument('--nsims', type=int, default=100_000, help='Global Nsims')
parser.add_argument('--nbins', type=int, default=100, help='Global Nbins to simulate over')
parser.add_argument('--bounds', type=int, default=5, help='Global bounds to run simulator on')
# strings
parser.add_argument('--mode', type=str, default='complex', help='Global Simulator mode')
parser.add_argument('--det', type=str, default='det', help='Are simulators deterministic or stochastic? det or stoch.')
#bools
parser.add_argument('--pvebounds', action='store_true', help='Positive bounds? Uniform by default')
parser.add_argument('--nobkg', action='store_false', help='Removes Background mu')

args = parser.parse_args()

glob_sigma = args.sigma
glob_bkg = args.nobkg
glob_pve_bounds = args.pvebounds
glob_det = args.det
glob_mode = args.mode

Nsims = args.nsims
Nbins = args.nbins
train_bounds = args.bounds

simulator = Simulator_Additive(Nbins=Nbins, sigma=glob_sigma, bounds=train_bounds, 
                               fraction=0.2, bkg=glob_bkg, dtype=torch.float32, 
                               mode=glob_mode, pve_bounds=glob_pve_bounds, bump=glob_det)     
samples = simulator.sample(Nsims=Nsims)
obs = simulator.sample(1)

p_marker = 'p' if glob_pve_bounds == True else 'n'
b_marker = 'b' if glob_bkg == True else 'q'
d_marker = 'd' if glob_det == 'det' else 's'
# netid = p_marker+b_marker+s_marker+str(train_bounds)
netid = 'eMu-d_'+p_marker+b_marker+d_marker+str(train_bounds)
print(f'netid {netid}')

if not os.path.isdir('figs/'+netid):
    os.makedirs('figs/'+netid)

######### NEURAL NETOWORK CODE #########

### -- make network -- ###


from models.online_norm import OnlineStandardizingLayer
from models.resnet_1d import ResidualNet
from models.unet_1d import UNet1d 

# - snr - #

class Network_epsilon(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.logvariance = torch.nn.Parameter(torch.ones(Nbins)*5)
        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        self.mu_predictor = torch.nn.Sequential(
            torch.nn.Linear(Nbins, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, Nbins)
        )
    
    def mu(self, x):
        x = self.mu_predictor(x.unsqueeze(1)).squeeze(1)
        return x
                
    def epsilon(self, x):
        resd = x - self.mu(x)
        out = self.net(resd.unsqueeze(1)).squeeze(1) # x-net
        return out
    
    def snr(self, x):
        return self.epsilon(x) / self.logvariance.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        # Adaptive data generation
        ni = x['ni']
        
       # generate a noise block like x #
        x_shape = x['x0'].shape # [nsims,nbins]
        noise = torch.complex(torch.randn(x_shape).cuda(), torch.randn(x_shape).cuda()).cuda()
        norm_noise = torch.abs(noise)

        # generate a mu block like x #
        m, amp, sigma = [Nbins/2,3,20]
        grid = torch.arange(Nbins).cuda()
        mu = amp * torch.exp(-0.5 * ((grid - m) / sigma) ** 2)
        mu_block = (torch.ones(x_shape).cuda())*mu.unsqueeze(0)

        ###########################################
        epsilon_sim =  (2 * self.bounds() * torch.rand(x['x'].shape, 
                                                           device= x['x'].device, 
                                                           dtype= x['x'].dtype) - self.bounds()) * ni
        ###########################################
        
        # data =  x['x0'] + epsilon_sim * ni

        data = norm_noise+mu_block+epsilon_sim
        
        # net evaluation_m
        net_mu = self.mu(data)
        error_mu = (net_mu-mu)**2
        l_mu = error_mu / (self.logvariance.exp() + 1e-10) + self.logvariance 
        l_mu_return = l_mu.sum() * 0.5
        # net evaluation_e
        net_epsilon = self.epsilon(data)
        mask = ( x['ni'] != 0 )  
        squared_error_e = (net_epsilon - epsilon_sim)**2                                                  # [B, N_bins]
        l_e = squared_error_e / (self.logvariance.exp() + 1e-10) + self.logvariance                     # [B, N_bins]
        l_e_return = (l_e * mask.float()).sum() * 0.5

        return l_mu_return+l_e_return
    

# - bce - #

class Network_BCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = UNet1d(1, 1, sizes=(8, 16, 32, 64, 128))
        # self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 
        self.online_norm = OnlineStandardizingLayer((Nbins,), use_average_std=False) 

    def forward(self, x):
        data = x['x']
        x = self.net(data.unsqueeze(1)).squeeze(1)
        return x

##### LOAD FROM PATH #####

## - snr - ##

network_epsilon = Network_epsilon()
checkpoint = torch.load(f'networks/network_{netid}_complex', 
                        weights_only=False,map_location=torch.device('cpu'))
sd = checkpoint.state_dict()
new_state_dict = {}
for key in sd:
    new_key = key.replace('model.', '')  # Remove 'model.' prefix
    new_state_dict[new_key] = sd[key]
    
network_epsilon.load_state_dict(new_state_dict)
# network_epsilon.cuda().double().eval()
network_epsilon.to(dtype=torch.float32,device='mps').eval()

model = CustomLossModule_withBounds(network_epsilon, learning_rate=3e-3)
checkpoint = torch.load(f'networks/model_{netid}_complex', 
                        weights_only=False,map_location=torch.device('cpu'))
sd = checkpoint.state_dict()
for key in sd:
    new_key = key.replace('model.', '')  # Remove 'model.' prefix
    new_state_dict[new_key] = sd[key]
    
model.load_state_dict(sd)
# network_epsilon.cuda().double().eval()
model.to(dtype=torch.float32,device='mps').eval()

## -- bce -- ##

network_BCE = Network_BCE()
checkpoint = torch.load(f'networks/network_BCE_{netid}_complex', weights_only=False,map_location=torch.device('cpu'))
sd = checkpoint.state_dict()
new_state_dict = {}
for key in sd:
    new_key = key.replace('model.', '')  # Remove 'model.' prefix
    new_state_dict[new_key] = sd[key]
    
network_BCE.load_state_dict(new_state_dict)
# network_BCE.cuda().double().eval()
network_BCE.to(dtype=torch.float32,device='mps').eval()

model_BCE = BCELossModule(network_BCE, learning_rate=3e-3)
checkpoint = torch.load(f'networks/model_BCE_{netid}_complex', weights_only=False,map_location=torch.device('cpu'))
sd = checkpoint.state_dict()
# new_state_dict = {}
for key in sd:
    new_key = key.replace('model.', '')  # Remove 'model.' prefix
    new_state_dict[new_key] = sd[key]
    
model_BCE.load_state_dict(sd)
# network_BCE.cuda().double().eval()
model_BCE.to(dtype=torch.float32,device='mps').eval()

######## MC ON DATA #########

N_mc = 2e6
obs = simulator.sample(1)

ni = torch.eye(Nbins, dtype=obs['xi'].dtype)
variance = 1 / get_sigma_epsilon_inv2(ni)

batch_size = 2048*2
N_batch = int(N_mc / batch_size)
data_bin_H0 = []
res_bin_H0 = [] 
eps_bin_H0 = [] 
for _ in range(N_batch):
    mc_samples = simulator.sample(batch_size)
    data_bin_H0.append(mc_samples['x0'])
    res_bin_H0.append(mc_samples['x0']-mc_samples['mu'])
    eps_bin_H0.append(mc_samples['epsilon'])
    
data_bin_H0 = np.concatenate(data_bin_H0)
eps_bin_H0 = np.concatenate(eps_bin_H0)
res_bin_H0 = np.concatenate(res_bin_H0)

batch_size = 2048*2
N_batch = int(N_mc / batch_size)
ts_bin_H0_BCE = []
ts_bin_H1_BCE = []
eps_real_BCE = []
for _ in range(N_batch):
    mc_samples = simulator.sample(batch_size)
    ts_bin_H0_BCE.append(ts_sbi(mc_samples, model=0))
    ts_bin_H1_BCE.append(ts_sbi(mc_samples, model=1))
    eps_real_BCE.append(mc_samples['epsilon'])

ts_bin_H0_BCE = np.concatenate(ts_bin_H0_BCE)
ts_bin_H1_BCE = np.concatenate(ts_bin_H1_BCE)
eps_real_BCE = np.concatenate(eps_real_BCE)

ts_bin_H0_BCE.shape

####### GET DISTRIBUTIONS ########

def get_quantiles(indata, nsig:int, sigma_key=False):
    data = np.sort(indata)
    sigmas = np.arange(-nsig,nsig+1,1)
    indices = (len(data)*norm.cdf(sigmas)).astype(int)
    if not sigma_key:
        return data[indices]
    else:
        return np.array([data[indices],sigmas])
    
alldata = data_bin_H0.flatten()
allres = res_bin_H0.flatten()
mean = np.mean(allres)
quantiles = get_quantiles(allres, 3)
quantiles_long = get_quantiles(allres,5)

###### GET RAW DISTRIBUTION #######

fig, ax1 = pf.create_plot(size=(4,2))
bin = 42
ax1.hist(res_bin_H0[:,bin], bins=50, density=True, color=mycolors[0],edgecolor='black', label=f'Bin {bin}')

ax1.set_ylabel('Density')
ax1.yaxis.set_label_coords(-0.15,1)
ax1.set_ylim([0,0.69])
ax1.set_xlabel(r'$\tilde{d}(f)$')
ax1.xaxis.set_label_coords(1,-0.15)

ax1.legend()
ax1.set_xlim([0,4])

ax2 = fig.add_axes((0,1,1,1), sharex=ax1)
plt.setp(ax2.get_xticklabels(), visible=False)

ax2.hist(allres, bins=50, density=True, color='#ff004f',edgecolor='black', label=f'All Data')
ax2.set_ylim([0,0.69])
ax2.vlines(quantiles, ymax=100, ymin=-100, color='#77aca2', lw=3)
ax2.legend()

ax3 = fig.add_axes((1,1,1,1), sharey=ax2)
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.hist(allres, bins=np.logspace(-2,np.log10(4),50), density=True, color='#ff004f',edgecolor='black', label=f'All Data')
ax3.vlines(quantiles, ymax=100, ymin=-100, color='#77aca2', lw=3)
ax3.set_xscale('log')
ax3.set_xlim([0.03,4])

ax4 = fig.add_axes((1,0,1,1), sharex=ax3, sharey=ax1)
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.hist(res_bin_H0[:,bin], bins=np.logspace(-2,np.log10(4),50), density=True, color=mycolors[0],edgecolor='black', label=f'All Data')
ax4.vlines(quantiles, ymax=100, ymin=-100, color='#77aca2', lw=3)
ax4.set_xscale('log')
ax4.set_xlim([0.03,4])

ax2_top = ax2.secondary_xaxis('top')
ax2_top.set_xticks(quantiles)
ax2_top.set_xticklabels([-3,-2,1,0,1,2,3])
ax2_top.tick_params(axis='x', which='major', direction='out', length=8)
ax2_top.set_xlabel(r'Deviations [$\sigma$]')

ax3_top = ax3.secondary_xaxis('top')
ax3_top.set_xticks(quantiles)
ax3_top.set_xticklabels([-3,-2,1,0,1,2,3])
ax3_top.tick_params(axis='x', which='major', direction='out', length=8)
ax3_top.tick_params(axis='x', which='minor', direction='out', top=False)
ax3_top.set_xlabel(r'Deviations [$\sigma$]')

ax1.vlines(quantiles, ymax=100, ymin=-100, color='#77aca2', lw=3)

pf.fix_plot([ax1,ax2, ax3, ax4])
plt.tight_layout()
plt.savefig('figs/distribution.png', dpi=700, bbox_inches = 'tight')

######## GET DISTRIBUTION WITH MU #######

test = simulator.sample(1)
quantiles_long = np.array([7.1508466e-04, 7.9613253e-03, 5.1986761e-02,
       2.1462703e-01, 5.8794379e-01, 1.1776060e+00,
       1.9190179e+00, 2.7507384e+00, 3.6350725e+00,
       4.5491748e+00, 5.4850187e+00], dtype=np.float32)

pf.housestyle_rcparams()
fig, ax1 = pf.create_plot()

plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = fig.add_axes((0,-.3,1,0.3), sharex=ax1)
ax3 = fig.add_axes((1,-.3,0.2,0.3), sharey=ax2)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)

ax1.plot(test['xi'][0], label=r'$x_i$', color="#d931f3")
ax1.plot(test['x0'][0], label=r'$x_0$', color='#ff004f')
ax1.plot(test['mu'][0], label=r'$\mu$', color='black')
ax1.set_ylabel(r'$|\tilde{d}(f)|$')
ax1.set_ylim([0,8])
ax1.legend(loc='upper right')

resd = test['x0'][0]-test['mu'][0]
ax2.plot(resd, color='#ff004f')
ax2.set_xlabel(r'$f$')
ax2.set_ylabel(r'res ($x_0$)')
ax2.set_ylim([0,4.4])
grid = torch.linspace(0, 100, 100)
for i in range(1,6):
    ax1.fill_between(grid, quantiles_long[i]+test['mu'][0], quantiles_long[-i]+test['mu'][0],  color='#b0b0b0', alpha=0.1)
    ax2.fill_between(grid, quantiles_long[i], quantiles_long[-i],  color='#b0b0b0', alpha=0.1)
    ax3.fill_between(grid, quantiles_long[i], quantiles_long[-i],  color='#b0b0b0', alpha=0.1)

ax3.hist(resd, orientation='horizontal', bins=14, edgecolor='black', color='#ff004f', density=True)
ax3.set_xlim([0,1])

pf.fix_plot([ax1,ax2, ax3])
plt.tight_layout()
plt.savefig(f'figs/data_visu_{d_marker}{b_marker}.png', dpi=700, bbox_inches = 'tight')

#### MCMC WITH STATS ####

###### SNR #######  

### - SNR - ###

N_mc = 2e6

ni = torch.eye(Nbins, dtype=obs['xi'].dtype)
variance = 1 / get_sigma_epsilon_inv2(ni)

batch_size = 2048*2
N_batch = int(N_mc / batch_size)

ts_bin_H0_epsilon = []
for _ in range(N_batch):
    mc_samples = simulator.sample(batch_size)
    # ts_batch =  (network_epsilon.snr(mc_samples['x0'].cuda())**2).detach().cpu().numpy()
    ts_batch =  (network_epsilon.snr(mc_samples['x0'].to(device='mps', dtype=torch.float32))**2).detach().cpu().numpy()
    ts_bin_H0_epsilon.append(ts_batch)
    
ts_bin_H0_epsilon = np.concatenate(ts_bin_H0_epsilon)


### - BCE - ### 

N_mc = 2e6

ni = torch.eye(Nbins, dtype=obs['xi'].dtype)
variance = 1 / get_sigma_epsilon_inv2(ni)

batch_size = 2048*2
N_batch = int(N_mc / batch_size)
data_bin_H0 = []
res_bin_H0 = [] 
eps_bin_H0 = [] 
for _ in range(N_batch):
    mc_samples = simulator.sample(batch_size)
    data_bin_H0.append(mc_samples['x0'])
    res_bin_H0.append(mc_samples['x0']-mc_samples['mu'])
    eps_bin_H0.append(mc_samples['epsilon'])
    
data_bin_H0 = np.concatenate(data_bin_H0)
eps_bin_H0 = np.concatenate(eps_bin_H0)
res_bin_H0 = np.concatenate(res_bin_H0)

##### PVALUES ######

# snr #

ni = torch.eye(Nbins, dtype=obs['xi'].dtype)
variance = 1 / get_sigma_epsilon_inv2(ni)

batch_size = 2048*2
N_batch = int(N_mc / batch_size)
data_bin_H0 = []
eps_bin_H0 = []
network_pvalue = []
for _ in range(N_batch):
    mc_samples = simulator.sample(batch_size)
    data_bin_H0.append(mc_samples['x0'])
    eps_bin_H0.append(mc_samples['epsilon'])
    network_pvalue.append(network_epsilon.snr(mc_samples['xi'].to(device='mps', dtype=torch.float32)).detach().cpu().numpy()**2)
    
data_bin_H0 = np.concatenate(data_bin_H0)
eps_bin_H0 = np.concatenate(eps_bin_H0)
network_pvalue = np.concatenate(network_pvalue)

N_mc, num_bins = ts_bin_H0_epsilon.shape
ts_bin_flat = ts_bin_H0_epsilon.reshape(N_mc, num_bins)
means = ts_bin_flat.mean(axis=0)  # Shape: [num_bins]
ts_centered = ts_bin_flat - means  # Shape: [N_mc, num_bins]

# Sort the centered data along N_mc axis
sorted_ts = np.sort(ts_centered, axis=0)  # Shape: [N_mc, num_bins]
# Compute ranks for all values
ranks = np.argsort(np.argsort(-ts_centered, axis=0), axis=0)  # Ranks in descending order
# Compute p-values
p_values = (ranks + 1) / N_mc
# Reshape back to [N_mc, num_bins]
pv_bin_H0 = p_values.reshape(N_mc, num_bins)  

##### PARITY PLOTS #### SNR NETWORK PARITY

pf.housestyle_rcparams()
n = 50

obs = simulator.sample(1)  
delta_x = obs['xi']

ni = torch.eye(Nbins, dtype=obs['xi'].dtype)
epsilon_nn_obs = network_epsilon.epsilon(delta_x.to(dtype=torch.float32,device='mps')).detach().cpu().numpy().squeeze(0)
variance_nn_obs = network_epsilon.logvariance.exp().detach().cpu().numpy()
snr_nn_obs = network_epsilon.snr(delta_x.to(dtype=torch.float32,device='mps')).detach().cpu().numpy().squeeze(0)
# epsilon_nn_obs = network_epsilon.epsilon(delta_x.cuda()).detach().cpu().numpy().squeeze(0)
# variance_nn_obs = network_epsilon.logvariance.exp().detach().cpu().numpy()
# snr_nn_obs = network_epsilon.snr(delta_x.cuda()).detach().cpu().numpy().squeeze(0)
epsilon_obs = get_epsilon(delta_x, ni).squeeze(0)
variance_obs = 1 / get_sigma_epsilon_inv2(ni)
snr_obs = get_snr(delta_x, ni).squeeze(0)

epsilon_nn_obs = network_epsilon.epsilon(delta_x.to(dtype=torch.float32,device='mps')).detach().cpu().numpy().squeeze(0)
variance_nn_obs = network_epsilon.logvariance.exp().detach().cpu().numpy()
snr_nn_obs = network_epsilon.snr(delta_x.to(dtype=torch.float32,device='mps')).detach().cpu().numpy().squeeze(0)

# For each simultion of the n ones, compute analytical quantities
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))

fig,ax1 = pf.create_plot(size=(3,3))
ax2 = fig.add_axes((1.5,0,1,1))
ax3 = fig.add_axes((2.5,0,0.5,1), sharey=ax2)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

axs = [ax1,ax2,ax3]
axs[0].scatter(x=epsilon_obs, y=epsilon_nn_obs, c='#ff004f', s=30, marker='o',linewidths=0.4, alpha=0.7, edgecolor='black')
axs[1].scatter(x=variance_obs, y=variance_nn_obs, c='#ff004f', s=30, marker='o',linewidths=0.4, alpha=0.7, edgecolor='black')
axs[2].hist(variance_nn_obs, orientation='horizontal', bins=20, color='white', edgecolor='black')
    
idx = torch.where(obs['ni']==1)[1]
axs[0].scatter(x=epsilon_obs[idx], y=epsilon_nn_obs[idx], c='#77aca2', s=50, marker='x',linewidths=2)
axs[1].scatter(x=variance_obs[idx], y=variance_nn_obs[idx], c='#77aca2', s=50, marker='x',linewidths=2)


axs[0].set_xlabel(r'$\epsilon$ true')
axs[0].set_ylabel(r'$\epsilon$ NN')
axs[1].set_xlabel(r'$\sigma^2$ true')
axs[1].set_ylabel(r'$\sigma^2$ NN')
plt.tight_layout()

pf.fix_plot(axs)

plt.tight_layout()
plt.savefig(f'figs/{netid}/parity.png', dpi=700, bbox_inches = 'tight')

##### PARITY PLOTS ##### SNR MASS PARITY

fig,ax1 = pf.create_plot(size=(3.5,3))
ax2 = fig.add_axes((1.2,0,1,1))

ax1.scatter(eps_bin_H0[:10000], network_pvalue[:10000],s=.5, alpha=0.1, color=folly)
ax2.scatter(eps_bin_H0[:10000], network_pvalue[:10000],s=.5, alpha=0.1, color=folly)

x = np.linspace(0,8,100)
ax1.plot(x, x**2,lw=5, color='black')

ax1.set_xlabel(r'$\epsilon$ Analytical')
ax2.set_xlabel(r'$\epsilon$ Analytical')
ax1.set_ylabel(r'$\mathrm{SNR}_i^2$')
ax1.set_xlim([0,5])
ax1.set_ylim([0,40])
pf.fix_plot([ax1,ax2])
plt.tight_layout()
plt.savefig(f'figs/{netid}/preditction_distribution.png', dpi=700, bbox_inches = 'tight')

###### PARITY PLOTS ##### BCE MASS PARITY

fig,ax1 = pf.create_plot(size=(3.5,3))
ax3 = fig.add_axes((1.2,0,1,1))

ax1.scatter(eps_real_BCE[:10000], ts_bin_H1_BCE[:10000],s=.8, alpha=0.1, color=mycolors[0])
ax1.plot(np.linspace(0,5,100),np.linspace(0,5,100)**2, color='white', lw=5)

resd = ts_bin_H1_BCE[:10000] - (eps_real_BCE[:10000])**2

ax3.scatter(eps_real_BCE[:10000], ts_bin_H1_BCE[:10000],s=.8, alpha=0.1, color=mycolors[0])

ax1.set_xlabel(r'$\epsilon$ Analytical')
ax3.set_xlabel(r'$\epsilon$ Analytical')
ax1.set_ylabel(r'$\mathrm{t}_i$')
ax1.set_xlim([0,5])
ax1.set_ylim([0,30])
pf.fix_plot([ax1,ax2,ax3])
plt.tight_layout()
plt.savefig(f'figs/{netid}/BCE_parity.png', dpi=700, bbox_inches = 'tight')

##### TEST STATISTIC STUFF ##### SNR 

pf.housestyle_rcparams()
grid = np.linspace(0, 10, 100) # 1 df, adjust
chi2 = scipy.stats.chi2.pdf(grid, df=1, loc=0)

fig, axs = plt.subplots(1, 4, figsize=(15, 4), dpi=200)
for i in range(4):
    bin = torch.randint(Nbins, (1,))
    ts_bin_i = ts_bin_H0_epsilon[:, bin]
    bins = np.linspace(0, 10, 100)
    axs[i].hist(ts_bin_i, bins=bins, density=True, color='#ff004f', alpha=0.5)    
    axs[i].set_xlabel(f'$t_i$, bin {int(bin)}')
    axs[i].set_xlim(-1, 7)
    axs[i].plot(grid, chi2, c='k', label=r'$\chi^2$ with df=1')
axs[0].set_ylabel('Freq Density')

pf.fix_plot(axs)

plt.tight_layout()
plt.savefig(f'figs/{netid}/histogram_complex.png', dpi=300)

##### TEST STATISTIC STUFF ##### SNR SUM

DOF = Nbins

ts_sum_H0_epsilon = ts_bin_H0_epsilon.sum(axis=1)
ts_sum_H0_epsilon_mean = ts_sum_H0_epsilon.mean()
ts_sum_H0_epsilon = ts_sum_H0_epsilon - ts_sum_H0_epsilon_mean

fig, ax1 = pf.create_plot()

bins = np.linspace(ts_sum_H0_epsilon.min(), ts_sum_H0_epsilon.max(), 100)
ax1.plot(bins, scipy.stats.chi2.pdf(bins, df=DOF, loc=-DOF), c='k', label='Chi2 with df=Nbins')
ax1.hist(ts_sum_H0_epsilon, bins=bins, density=True, color='#ff004f', label='Sum $t(x|H_0)$ samples')
ax1.hist(ts_sum_H0_epsilon, bins=bins, density=True, color='black', histtype='step')
ax1.legend(loc='best', fontsize=12, labelspacing=0.1)

plt.tight_layout()
pf.fix_plot([ax1])
plt.savefig(f'figs/{netid}/histogram.png', dpi=700, bbox_inches = 'tight')

##### TEST STATISTIC STUFF ##### BCE 

grid = np.linspace(0, 10, 100) # 1 df, adjust
chi2 = scipy.stats.chi2.pdf(grid, df=1, loc=0)

fig, axs = plt.subplots(1, 4, figsize=(13, 3), dpi=200)
res = 50

for i in range(4):
    bin = np.random.randint(0,100)
    m = np.mean(ts_bin_H0_BCE[:,bin])
    axs[i].hist(ts_bin_H0_BCE[:,bin]-m, density=True, bins=np.linspace(0,10,res), color=mycolors[0], alpha=0.6)
    axs[i].hist(ts_bin_H0_BCE[:,bin]-m, density=True, bins=np.linspace(0,10,res), color='black', histtype='step')
    axs[i].set_xlim([0,10])
    axs[i].plot(grid,chi2, color='black')
    axs[i].set_ylim([0,0.5])
    axs[i].set_xlabel(f'$t_i$ bin {bin}')


pf.fix_plot(axs)
plt.tight_layout()
plt.savefig(f'figs/{netid}/BCE_tis.png', dpi=700, bbox_inches = 'tight')

##### TEST STATISTIC STUFF ##### BCE SUM

DOF = Nbins - 3 # 3 parameters

ts_sum_H0_BCE = ts_bin_H0_BCE.sum(axis=1)
ts_sum_H0_BCE_mean = ts_sum_H0_BCE.mean()
ts_sum_H0_BCE = ts_sum_H0_BCE - ts_sum_H0_BCE_mean

fig,ax1=pf.create_plot()

bins = np.linspace(ts_sum_H0_BCE.min(), ts_sum_H0_BCE.max(), 100)
ax1.plot(bins, scipy.stats.chi2.pdf(bins, df=DOF, loc=-DOF), c='k', label='Chi2 with df=Nbins')

ax1.hist(ts_sum_H0_BCE, bins=bins, density=True, color=mycolors[0], label='Sum $t(x|H_0)$ samples')
ax1.hist(ts_sum_H0_BCE, bins=bins, density=True, color='black', histtype='step')

ax1.legend(loc='best', fontsize=12, labelspacing=0.1)

pf.fix_plot([ax1])

plt.tight_layout()
plt.savefig(f'figs/{netid}/BCE_hist.png', dpi=700, bbox_inches = 'tight')

############ P VALUES STUFF ##################


pf.housestyle_rcparams()
bins_pairs = [(0, 1), (0, 10), (0, 90)]

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for i, (bin1, bin2) in enumerate(bins_pairs):
    p_values_res1 = pv_bin_H0[:, bin1]
    p_values_res2 = pv_bin_H0[:, bin2]

    axs[i].scatter(p_values_res1, p_values_res2, alpha=0.5, s=1, color=folly)
    axs[i].set_xlabel(f'P-values bin {bin1}')
    axs[i].set_ylabel(f'P-values bin {bin2}')
    axs[i].set_xlim(1e-6, 1)
    axs[i].set_ylim(1e-6, 1)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].grid(True)

pf.fix_plot(axs)
plt.tight_layout()
plt.savefig(f'figs/{netid}/correlations.png', dpi=700, bbox_inches = 'tight')

pf.housestyle_rcparams()
### - CENTER AND RANK TS_SUM UNDER H0 - ###
# Center the test statistic (e.g., chi-squared-like) under H0 by subtracting the mean
means = ts_sum_H0_epsilon.mean(axis=0)  # Mean of summed test statistic across MC samples
ts_centered = ts_sum_H0_epsilon - means  # Centered statistic

# Sort and rank the centered values to compute empirical p-values
sorted_ts = np.sort(ts_centered, axis=0)
ranks = np.argsort(np.argsort(-ts_centered, axis=0))  # Descending ranks
p_values = (ranks + 1) / N_mc  # Empirical p-values

pv_sum_H0 = p_values  # Final p-values for the sum test
pv_sum_H0.shape  # Check shape, should be (N_mc,)

### - PLOT HISTOGRAM OF EMPIRICAL P-VALUES - ###
fig,ax1=pf.create_plot(size=(3,3))
ax1.hist(pv_sum_H0, bins=50, alpha=0.7, color=folly)
ax1.set_xlabel('Empirical p-value')
ax1.set_ylabel('Frequency')
ax1.set_title("P-value Distribution\n"+r"(Sum Statistic under $H_0$)")

# This shows the distribution of the p-values under H0 for the sum test

############### THIS COULD BE A MISTAKE REPEAT PLOT ######################
### - PLOT MINIMUM P-VALUES FROM SUM TEST - ###
Nmc = pv_bin_H0.shape[0]
min_pv_bin_H0_BCE = np.min(pv_bin_H0.reshape(Nmc, -1), axis=1)
min_pv_sum_H0_epsilon = np.min(pv_sum_H0.reshape(Nmc, -1), axis=1)
min_pv_bin_H0_epsilon = np.min(pv_bin_H0.reshape(Nmc, -1), axis=1)


ax2 = fig.add_axes((1.3,0,1,1))
ax2.hist(min_pv_sum_H0_epsilon, bins=50, alpha=0.7, color=folly)
ax2.set_xlabel('Minimum empirical\np-value (sum tests)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of\nMinimum P-values\n'+r'(Sum Test, under $H_0$)')
#########################################################################

### - COMBINE BIN-WISE AND SUM P-VALUES - ###
Nmc = pv_bin_H0.shape[0]
pv_all_H0 = np.concatenate([
    pv_bin_H0.reshape(Nmc, -1),     # Bin-wise p-values
    pv_sum_H0.reshape(Nmc, -1)      # Sum p-values
], axis=1)  # Combined shape: [N_mc, N_bins + 1]

# print(pv_all_H0.shape)  # Sanity check

### - PLOT MINIMUM P-VALUES ACROSS ALL TESTS - ###
min_pv_all_H0_epsilon = np.min(pv_all_H0, axis=1)  # Take minimum p-value across all bins + sum

ax3 = fig.add_axes((2.6,0,1,1))
ax3.hist(min_pv_all_H0_epsilon, bins=50, alpha=0.7, color=folly)
ax3.set_xlabel('Minimum empirical\np-value (all tests)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of\nMinimum P-values\n'+'(All Tests under $H_0$)')

pf.fix_plot([ax1,ax2,ax3])
pf.fix_plot([ax1])
plt.savefig(f'figs/{netid}/pvaluedists.png', dpi=700, bbox_inches = 'tight')

#### BCE #####################################################

N_mc, num_bins = ts_bin_H0_BCE.shape
ts_bin_flat = ts_bin_H0_BCE.reshape(N_mc, num_bins)
means = ts_bin_flat.mean(axis=0)  # Shape: [num_bins]
ts_centered = ts_bin_flat - means  # Shape: [N_mc, num_bins]

sorted_ts = np.sort(ts_centered, axis=0)  # Shape: [N_mc, num_bins]
ranks = np.argsort(np.argsort(-ts_centered, axis=0), axis=0)  # Ranks in descending order
p_values = (ranks + 1) / N_mc
pv_bin_H0 = p_values.reshape(N_mc, num_bins)   

pf.housestyle_rcparams()

### - NOW PLOT SOME PAIRS OF BINS - ###
bins_pairs = [(0, 1), (0, 10), (0, 90)]

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for i, (bin1, bin2) in enumerate(bins_pairs):
    p_values_res1 = pv_bin_H0[:, bin1]
    p_values_res2 = pv_bin_H0[:, bin2]

    axs[i].scatter(p_values_res1, p_values_res2, alpha=0.5, s=1, color=mycolors[0])
    axs[i].set_xlabel(f'P-values bin {bin1}')
    axs[i].set_ylabel(f'P-values bin {bin2}')
    axs[i].set_xlim(1e-6, 1)
    axs[i].set_ylim(1e-6, 1)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].grid(True)
pf.fix_plot(axs)
    
plt.tight_layout()
plt.savefig(f'figs/{netid}/BCE_correlations.png', dpi=700, bbox_inches = 'tight')

### - CENTER AND RANK TS_SUM UNDER H0 - ###
# Center the test statistic (e.g., chi-squared-like) under H0 by subtracting the mean
means = ts_sum_H0_BCE.mean(axis=0)  # Mean of summed test statistic across MC samples
ts_centered = ts_sum_H0_BCE - means  # Centered statistic

# Sort and rank the centered values to compute empirical p-values
sorted_ts = np.sort(ts_centered, axis=0)
ranks = np.argsort(np.argsort(-ts_centered, axis=0))  # Descending ranks
p_values = (ranks + 1) / N_mc  # Empirical p-values

pv_sum_H0 = p_values  # Final p-values for the sum test
pv_sum_H0.shape  # Check shape, should be (N_mc,)

### - PLOT HISTOGRAM OF EMPIRICAL P-VALUES - ###
fig,ax1=pf.create_plot(size=(3,3))
ax1.hist(pv_sum_H0, bins=50, alpha=0.7, color=mycolors[0])
ax1.set_xlabel('Empirical p-value')
ax1.set_ylabel('Frequency')
ax1.set_title("P-value Distribution\n"+r"(Sum Statistic under $H_0$)")

# This shows the distribution of the p-values under H0 for the sum test

############### THIS COULD BE A MISTAKE REPEAT PLOT ######################
### - PLOT MINIMUM P-VALUES FROM SUM TEST - ###
min_pv_sum_H0_BCE = np.min(pv_sum_H0.reshape(Nmc, -1), axis=1)

ax2 = fig.add_axes((1.3,0,1,1))
ax2.hist(min_pv_sum_H0_epsilon, bins=50, alpha=0.7, color=mycolors[0])
ax2.set_xlabel('Minimum empirical\np-value (sum tests)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of\nMinimum P-values\n'+r'(Sum Test, under $H_0$)')
#########################################################################

### - COMBINE BIN-WISE AND SUM P-VALUES - ###
Nmc = pv_bin_H0.shape[0]
pv_all_H0 = np.concatenate([
    pv_bin_H0.reshape(Nmc, -1),     # Bin-wise p-values
    pv_sum_H0.reshape(Nmc, -1)      # Sum p-values
], axis=1)  # Combined shape: [N_mc, N_bins + 1]

# print(pv_all_H0.shape)  # Sanity check

### - PLOT MINIMUM P-VALUES ACROSS ALL TESTS - ###
min_pv_all_H0_BCE = np.min(pv_all_H0, axis=1)  # Take minimum p-value across all bins + sum

ax3 = fig.add_axes((2.6,0,1,1))
ax3.hist(min_pv_all_H0_BCE, bins=50, alpha=0.7, color=mycolors[0])
ax3.hist(min_pv_all_H0_BCE, bins=50, alpha=0.7, color='black', histtype='step')
ax3.set_xlabel('Minimum empirical\np-value (all tests)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of\nMinimum P-values\n'+'(All Tests under $H_0$)')

pf.fix_plot([ax1,ax2,ax3])
plt.tight_layout()
plt.savefig(f'figs/{netid}/BCE_distributions.png', dpi=700, bbox_inches = 'tight')


########################################################################################
###################    FINAL COMPARISONS    ###################
########################################################################################

def analyse_obs_epsilon(obs):
    
    target = obs['xi']
    
    # Evaluate epsilon and SNR^2 test statistic from NN
    epsilon_nn = network_epsilon.epsilon(target.to(device='mps', dtype=torch.float32)).detach().cpu().numpy().squeeze(0)       #[len(correlation_scales), Nbins]
    # epsilon_nn = network_epsilon.epsilon(target.cuda()).detach().cpu().numpy().squeeze(0)       #[len(correlation_scales), Nbins]

    variance_nn = network_epsilon.logvariance.exp().detach().cpu().numpy()                            #[len(correlation_scales), Nbins]
    # variance_nn = network_epsilon.logvariance.exp().detach().cpu().numpy()                            #[len(correlation_scales), Nbins]

    snr2_nn = network_epsilon.snr(target.to(device='mps', dtype=torch.float32)).detach().cpu().numpy().squeeze(0)**2            #[len(correlation_scales), Nbins]
    # snr2_nn = network_epsilon.snr(target.cuda()).detach().cpu().numpy().squeeze(0)**2            #[len(correlation_scales), Nbins]

    ts_sum_nn = snr2_nn.sum()-ts_sum_H0_epsilon_mean
    p_sum_nn = t_to_pvalue_empirical(ts_sum_nn, ts_sum_H0_epsilon)   

    
    # Compute analytical epsilon and SNR^2 test statistic
    ni_temp = torch.eye(Nbins, dtype=torch.float32)
    fit = best_fit(obs['xi'][0], simulator)
    delta_x = (obs['xi'] - fit).to(dtype=torch.float32)
    epsilon_analytical = get_epsilon(delta_x, ni_temp).squeeze(0)
    snr2_analytical = get_snr(delta_x, ni_temp).squeeze(0)**2
    ts_sum_analytical = snr2_analytical.sum() #((obs['xi'])**2/glob_sigma**2).sum()
    p_sum_analytical = t_to_pvalue(ts_sum_analytical, DOF)

    # Compute localized p-values
    _p_nn, _p_analytical = [], []
    for idx, ts_bin in enumerate(snr2_nn):
        ts_bin_i = ts_bin_H0_epsilon[:, idx]
        m = ts_bin_i.mean()
        ts0_ref = ts_bin_i - m
        ts_obs = (ts_bin-m)
        _p_nn.append(t_to_pvalue_empirical(ts_obs, ts0_ref))  
        _p_analytical.append(t_to_pvalue(snr2_analytical[idx], 1)) # 1 Nbins per bin
    p_nn = np.array(_p_nn) 
    p_analytical = np.array(_p_analytical)


    # Compute global p-values
    obs_min_pv_bin = p_nn.reshape(-1).min()
    obs_min_pv_sum = p_sum_nn
    pv_all_obs = np.concatenate([
        p_nn.reshape(-1),  # Shape: [Nbins]
        torch.tensor([p_sum_nn])  # Shape: [1]
    ], axis=0)  # Combined shape: [num_total_tests]
    obs_min_pv_all = pv_all_obs.min()

    p_glob_bin = np.mean(min_pv_bin_H0_epsilon <= obs_min_pv_bin)
    p_glob_all = np.mean(min_pv_all_H0_epsilon <= obs_min_pv_all)

    p_glob_bin, p_glob_all

    return epsilon_nn, epsilon_analytical, variance_nn, snr2_nn, snr2_analytical, p_nn, p_analytical, p_sum_nn, p_sum_analytical, p_glob_all


def analyse_obs_BCE(obs):
    
    target = obs['xi']
    
    ts_bin_obs = ts_sbi(obs, model=1)
    ts_bin_analytical = (((obs['xi']- best_fit(obs['xi'][0], simulator))**2)/glob_sigma**2)[0]-1
    ts_sum_obs = ts_bin_obs.sum()-ts_sum_H0_BCE_mean
    ts_sum_obs_analytical = (((obs['xi']- best_fit(obs['xi'][0], simulator))**2)/glob_sigma**2).sum()-DOF
    _p_nn, _p_analytical = [], []
    for idx, ts_bin in enumerate(ts_bin_obs):
        ts_bin_i = ts_bin_H0_BCE[:, idx]
        m = ts_bin_i.mean()
        ts0_ref = ts_bin_i - m
        ts_obs = (ts_bin-m)
        _p_nn.append(t_to_pvalue_empirical(ts_obs, ts0_ref))
        _p_analytical.append(t_to_pvalue(ts_bin_analytical[idx]+1, 1))
    p_nn = np.array(_p_nn)
    p_analytical = np.array(_p_analytical)
    p_sum_nn = t_to_pvalue_empirical(ts_sum_obs, ts_sum_H0_BCE)   
    p_sum_analytical = t_to_pvalue(ts_sum_obs_analytical+DOF, DOF)

    # Compute global p-values
    obs_min_pv_bin = p_nn.reshape(-1).min()
    obs_min_pv_sum = p_sum_nn
    pv_all_obs = np.concatenate([
        p_nn.reshape(-1),  # Shape: [Nbins]
        torch.tensor([p_sum_nn])  # Shape: [1]
    ], axis=0)  # Combined shape: [num_total_tests]
    obs_min_pv_all = pv_all_obs.min()

    p_glob_bin = np.mean(min_pv_bin_H0_BCE <= obs_min_pv_bin)
    p_glob_all = np.mean(min_pv_all_H0_BCE <= obs_min_pv_all)

    p_glob_bin, p_glob_all

    return ts_bin_obs, ts_bin_analytical, p_nn, p_analytical, p_sum_nn, p_sum_analytical, p_glob_all


def plot_analysis_BCE(obs, ts_bin_obs, ts_bin_analytical, p_nn, p_analytical, p_sum_nn, p_sum_analytical, p_glob_all):
    
    # Figure
    fig = plt.figure(figsize=(10, 8), dpi=200)
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[2, 1])

    ### FIRST PART
    # First row: ax1 spans both columns
    ax1 = fig.add_subplot(gs[0, 0])
    xi = obs['xi'][0]
    ni = obs['ni'][0] 
    dist = obs['xi'][0] - obs['x0'][0]
    grid = torch.linspace(0, Nbins, Nbins)
    ax1.plot(grid, obs['mu'][0], color='k', label=r"$\mu_{\mathrm{sim}}$")
    ax1.fill_between(grid, obs['mu'][0]-1, obs['mu'][0]+1,  color='#b0b0b0', alpha=0.1)
    ax1.fill_between(grid, obs['mu'][0]-2, obs['mu'][0]+2,  color='#b0b0b0', alpha=0.2)
    ax1.fill_between(grid, obs['mu'][0]-3, obs['mu'][0]+3,  color='#b0b0b0', alpha=0.3)
    ax1.scatter(grid, xi, c='k', marker='x', s=6)
    ax1.plot(grid, obs['mu'][0]+dist, color=mycolors[1], label=r"$\mu_{\mathrm{dist}}$")
    ax1.set_ylabel(r"$x_\mathrm{obs}$", labelpad=1.5)
    ax1.legend(fontsize=13, loc='best', labelspacing=0.1)
    ax1.set_ylim(-6.5, 6.5)
    ax1.set_xticks([])
    ax1.set_title("Data")

    # Second column: ax2 and ax3 in the first column
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(grid, p_nn, c=mycolors[1], label=r"$\mathrm{SBI}$")
    ax2.semilogy(grid, p_analytical, c='k', ls='dotted', label=r"analytical")
    ax2.set_ylabel(r"$\mathrm{p}_\mathrm{obs}$", labelpad=1.5)
    ax2.legend(loc='best',  fontsize=13)
    ax2.set_ylim(1/(N_mc*5), 1)
    ax2.set_xticks([])
    ax2.set_title("Anomaly detection")
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(grid, ts_bin_obs-ts_bin_obs.min(), c=mycolors[1], label=r"$\mathrm{SBI}$")
    ax3.plot(grid, ts_bin_analytical - ts_bin_analytical.min(), c='k', ls='dotted',label=r"analytical")
    ax3.set_ylabel(r"$t_\mathrm{obs}$")
    ax3.legend(loc='best',  fontsize=13)
    ax3.set_ylim(-.1, None)
    ax3.set_xticks([])
    ax3.set_title("")
    ax3.axhline(0, color='#b0b0b0', linestyle='--')


    # Second column: ax4 and ax6 in the second column

    # Bars
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_yscale('log')
    ax4.set_ylim(1/(N_mc*5), 1)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["SBI", "Analytical"])
    ax4.set_xlim(-0.5, 2 - 0.5)
    ax4.set_ylabel(r"$\mathrm{p}_\mathrm{sum}$")
    ax4.fill_between([-0.2, 0.2], 1, p_sum_nn, facecolor=mycolors[1], edgecolor=mycolors[1])
    ax4.fill_between([0.8, 1.2], 1, p_sum_analytical, alpha=0.2, edgecolor='k', facecolor='k', linestyle = 'dotted')
    ax4.set_title("Model validation")
    ax4.grid(True, axis='y')

    
    # Add on gs[0, 1] text reporting the three global p-values
    ax_text = fig.add_subplot(gs[0, 1])
    mantissa, exp = ('%.2e' % p_glob_all).split('e')
    exp = int(exp)
    textstr = r'$\mathrm{p}_{\mathrm{glob}}= %s \times 10^{%d}$' % (mantissa, exp)
    ax_text.text(0.5, 0.5, textstr, transform=ax_text.transAxes, fontsize=15,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(
                    facecolor=mygold,
                    edgecolor='none'  # Remove the border if not needed
                )
            )
    ax_text.axis('off')
    ax_text.set_title("Global p-value")
    pf.fix_plot([ax1,ax2,ax3,ax4])

    plt.tight_layout();

import matplotlib.gridspec as gridspec
pf.housestyle_rcparams()

def plot_together_new(
    obs, 
    ts_bin_obs, ts_bin_analytical, p_nn_BCE, p_analytical_BCE, p_sum_nn_BCE, p_sum_analytical_BCE, p_glob_all_BCE,
    snr2_nn, snr2_analytical, p_nn_epsilon, p_analytical_epsilon, p_sum_nn_epsilon, p_sum_analytical_epsilon, p_glob_all_epsilon
    ):
    
    pf.housestyle_rcparams()
    # fig = plt.figure(figsize=(10, 8), dpi=200)
    # gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[2, 1])

    ### FIRST PART
    # First row: ax1 spans both columns
    # gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], width_ratios=[3, 1], wspace=0.0)
    # ax1 = fig.add_subplot(gs0[0])
    # ax_hist = fig.add_subplot(gs0[1], sharey=ax1)

    spacing = 0.4
    fig, ax1 = pf.create_plot(size=(6,2))
    ax_hist = fig.add_axes((1,0,0.2,1), sharey=ax1)
    plt.setp(ax_hist.get_xticklabels(), visible=False)
    plt.setp(ax_hist.get_yticklabels(), visible=False)

    ax2 = fig.add_axes((0,-1-spacing,1,1),sharex=ax1)
    ax3 = fig.add_axes((0,2*(-1-spacing),1,1),sharex=ax1)

    ax4 = fig.add_axes((1+0.5*spacing,-1-spacing,0.4,1))
    ax6 = fig.add_axes((1+0.5*spacing,2*(-1-spacing),0.4,1))

    ax_text = fig.add_axes((1.1+0.5*spacing,0,0.3,1))
    
    mu = obs['mu'][0]
    x0 = obs['x0'][0]
    xi = obs['xi'][0]
    ni = obs['ni'][0] 
    dist = obs['xi'][0] - obs['x0'][0]
    grid = torch.linspace(0, Nbins, Nbins)
    ax1.plot(grid, obs['mu'][0], color='k', label=r"$\mu_{\mathrm{sim}}$", zorder=10)

    for i in range(1,6):
        ax1.fill_between(grid, (quantiles_long[i]+mu), 
                         (quantiles_long[-i]+mu),  color='#b0b0b0', alpha=0.1)
        ax_hist.fill_between(grid, quantiles_long[i], quantiles_long[-i],  color='#b0b0b0', alpha=0.1)


    ax1.scatter(grid, xi, c='k', marker='x', s=6)
    ax1.plot(grid, obs['mu'][0]+dist, color="#ff7b5a", label=r"$\mu_{\mathrm{dist}}$")
    ax1.set_ylabel(r"$x_\mathrm{obs}$", labelpad=1.5)
    ax1.legend(bbox_to_anchor=(0.05, 1), fontsize=13, loc='upper left', labelspacing=0.1)
    ax1.set_ylim(0, 8.5)

    mu_label = 'Deterministic' if glob_det=='det' else 'Stochastic'
    eps_label = 'Positive' if glob_pve_bounds else 'Symmetric'

    ax1.set_title(f"Data: Training {mu_label} "+r'$\mu$'+f' On {eps_label} '+r'$\epsilon$')

    ####### NEW HISTOGRAM #########
    ax_hist.hist(x0-mu, orientation='horizontal', color='white', 
                 edgecolor='black', density=True, bins=np.linspace(0,6,17))
    ax_hist.set_xlim([0,0.7])
    

    # Second column: ax2 and ax3 in the first column
    # ax2 = fig.add_subplot(gs[1, 0])

    # ax2.semilogy(grid, p_nn_BCE, c=mycolors[1], label=r"$\mathrm{BCE}$")
    # ax2.semilogy(grid, p_nn_epsilon, c=mycolors[2], label=r"$\mathrm{SNR}$")

    ax2.semilogy(grid, p_nn_BCE, c=mycolors[0], label=r"$\mathrm{BCE}$")
    ax2.semilogy(grid, p_nn_epsilon, c='#ff004f', label=r"$\mathrm{SNR}$")
  
    # ax2.semilogy(grid, p_analytical_BCE, c='k', ls='dotted', label=r"Analytical")
    # ax2.semilogy(grid, p_analytical_epsilon, c='blue', ls='dotted', label=r"analytical")
    ax2.set_ylabel(r"$\mathrm{p}_\mathrm{obs}$", labelpad=1.5)
    ax2.set_ylim(1/(N_mc*5), 1)
    ax2.set_title("Anomaly detection")
    ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax2.yaxis.set_minor_locator(ticker.NullLocator())
    ax2.grid(True, axis='y', which='major')
    ax2.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])  # Specify tick positions
    ax2.set_yticklabels([r"$10^{-1}$", None , r"$10^{-3}$", None, r"$10^{-5}$", None])  # Customize tick labels
    # ax2.legend(loc='best', fontsize=13, frameon=True, facecolor='white', framealpha=1, fancybox=True, edgecolor='white')

    # ax3 = fig.add_subplot(gs[2, 0])

    ax3.plot(grid, ts_bin_obs-ts_bin_obs.min(), c=mycolors[0], label=r"$\mathrm{BCE}$")
    ax3.plot(grid, snr2_nn-snr2_nn.min(), c='#ff004f', label=r"$\mathrm{SNR}$")

    # ax3.plot(grid, ts_bin_analytical - ts_bin_analytical.min(), c='k', ls='dotted',label=r"Analytical")
    # ax3.plot(grid, snr2_analytical - snr2_analytical.min(), c='blue', ls='dotted',label=r"analytical")
    ax3.set_ylabel(r"$t_\mathrm{obs}$")
    ax3.legend(loc='upper right', fontsize=13)
    ax3.set_ylim(-.1, None)
    ax3.set_title(r"Localized $t_i$")
    # ax3.grid(True, axis='y', which='both')
    

    distortion_locations = grid[ni != 0]
    for plot_ax in [ax1, ax2, ax3]:
        for loc in distortion_locations:
            plot_ax.axvline(loc, color='black', lw=1, zorder=-10)



    # Second column: ax4 and ax6 in the second column

      # Bars
    # ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_yscale('log')
    ax4.set_ylim(1/(N_mc*5), 1)
    

    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["BCE", "SNR"])
    ax4.set_xlim(-0.5, 2 - 0.5)

    
    ax4.set_ylabel(r"$\mathrm{p}_\mathrm{sum}$")
    

    ax4.fill_between([-0.2, 0.2], 1, p_sum_nn_BCE, facecolor=mycolors[0], edgecolor='black')
    ax4.fill_between([0.8, 1.2], 1, p_sum_nn_epsilon, facecolor='#ff004f', edgecolor='black')

    ax4.set_title(r"Model validation")
    ax4.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax4.yaxis.set_minor_locator(ticker.NullLocator())
    ax4.grid(True, axis='y', which='major')
    ax4.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])  # Specify tick positions
    ax4.set_yticklabels([r"$10^{-1}$", None , r"$10^{-3}$", None, r"$10^{-5}$", None])  # Customize tick labels
    
    # SUM
    # ax6 = fig.add_subplot(gs[2, 1])
    bins = np.linspace(ts_sum_H0_BCE.min(), ts_sum_H0_BCE.max(), 100)

    ax6.hist(ts_sum_H0_BCE, bins=bins, density=True, color=mycolors[0], alpha=0.5, label='BCE')
    ax6.hist(ts_sum_H0_BCE, bins=bins, density=True, color=mycolors[0], histtype='step')
    ax6.hist(ts_sum_H0_epsilon, bins=bins, density=True, color='#ff004f', alpha=0.5, label='SNR')
    ax6.hist(ts_sum_H0_epsilon, bins=bins, density=True, color='#ff004f', histtype='step')

    ax6.plot(bins, scipy.stats.chi2.pdf(bins, df=DOF, loc=-DOF), c='k', linestyle = 'dotted', label='Analytical')
    ax6.set_xlabel(r"$t_\mathrm{sum}$ $\mid H_0$")
    ax6.set_title(r"$\chi^2$")
    # ax6.legend(fontsize=13, loc='best', labelspacing=0.1, frameon=True, facecolor='white', framealpha=1, fancybox=True, edgecolor='white')
    
    # Add on gs[0, 1] text reporting the three global p-values
    # ax_text = fig.add_subplot(gs[0, 1])
    mantissa, exp = ('%.2e' % p_glob_all_BCE).split('e')
    exp = int(exp)
    textstr = r'$\mathrm{p}_{\mathrm{glob, BCE}}= %s \times 10^{%d}$' % (mantissa, exp)
    ax_text.text(0.5, 0.7, textstr, transform=ax_text.transAxes, fontsize=15,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(
                    facecolor=mygold,
                    edgecolor='none'  # Remove the border if not needed
                )
            )
    mantissa, exp = ('%.2e' % p_glob_all_epsilon).split('e')
    exp = int(exp)
    textstr = r'$\mathrm{p}_{\mathrm{glob, SNR}}= %s \times 10^{%d}$' % (mantissa, exp)
    ax_text.text(0.5, 0.4, textstr, transform=ax_text.transAxes, fontsize=15,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(
                    facecolor=mygold,
                    edgecolor='none'  # Remove the border if not needed
                )
            )
    ax_text.axis('off')
    ax_text.set_title("Global p-value")
    pf.fix_plot([ax1,ax2,ax3,ax4,ax6, ax_hist])

    plt.tight_layout()

bounds = [5,5,8]
frac = [0.01,0.1,0.05]

for i in range(3):
    simulator1 = Simulator_Additive(Nbins=Nbins, sigma=glob_sigma, bkg=glob_bkg, 
                                    bounds=bounds[i], fraction=frac[i], dtype=torch.float32, 
                                    mode=glob_mode, pve_bounds=glob_pve_bounds, bump=glob_det) 
    obs = simulator1.sample(1) 
        
    ts_bin_obs, ts_bin_analytical, p_nn_BCE, p_analytical_BCE, p_sum_nn_BCE, p_sum_analytical_BCE, p_glob_all_BCE = analyse_obs_BCE(obs)
    epsilon_nn, epsilon_analytical, variance_nn, snr2_nn, snr2_analytical, p_nn_epsilon, p_analytical_epsilon, p_sum_nn_epsilon, p_sum_analytical_epsilon, p_glob_all_epsilon = analyse_obs_epsilon(obs)

    plot_together_new(
        obs, 
        ts_bin_obs, ts_bin_analytical, p_nn_BCE, p_analytical_BCE, p_sum_nn_BCE, p_sum_analytical_BCE, p_glob_all_BCE,
        snr2_nn, snr2_analytical, p_nn_epsilon, p_analytical_epsilon, p_sum_nn_epsilon, p_sum_analytical_epsilon, p_glob_all_epsilon
        )

    plt.tight_layout()
    plt.savefig(f'figs/{netid}/complex_comparison_n{i}.png', dpi=700, bbox_inches = 'tight')

################# PART TWO ###########################

###### SET UP GRID ######
positions = torch.arange(0, Nbins, 1).to(dtype=simulator.dtype)
amplitudes = torch.linspace(-3, 10, 80).to(dtype=simulator.dtype)

position_grid, amplitude_grid = torch.meshgrid(positions,amplitudes)
b = {'x':amplitude_grid.T}

###### DO BCE STATS ######
a = ts_sbi(b)
s = get_snr2(b)
dat = [a,s]

fig, ax1 = pf.create_plot()
ax1.set_xlabel(r'$f$')
ax2 = fig.add_axes((1.05, 0,0.1,1))

ax3 = fig.add_axes((0, 1.1,1,1))
plt.setp(ax3.get_xticklabels(), visible=False)
ax4 = fig.add_axes((1.05, 1.1,0.1,1))

axs = [ax1,ax2,ax3,ax4]

mu_label = 'Deterministic' if glob_det=='det' else 'Stochastic'
eps_label = 'Positive' if glob_pve_bounds else 'Symmetric'
ax3.set_title(f"Data: Training {mu_label} "+r'$\mu$'+f' On {eps_label} '+r'$\epsilon$')


for q in range(2):
    mesh = axs[2*q].pcolormesh(position_grid.T, amplitude_grid.T, np.log10(dat[q]-np.min(dat[q])+0.1), cmap='magma_r')
    if q==0:
        fig.colorbar(mesh,cax=axs[2*q+1], shrink=0.8, label=r'$\mathrm{log}_{10}($t$_{i, \mathrm{BCE}})$')
    else:
        fig.colorbar(mesh,cax=axs[2*q+1], shrink=0.8, label=r'$\mathrm{log}_{10}($t$_{i, \mathrm{SNR}})$')

    for j in range(2):
        axs[2*q].plot(chop_middle(positions)[j], chop_middle(obs['mu'][0])[j], color='white', linewidth=3)
        for i in range(5):
            alp = .5+(i/8)
            # print(a)
            axs[2*q].plot(chop_middle(positions)[j], chop_middle(obs['mu'][0]+quantiles[-i])[j], color='white', alpha=alp)

    x = 47
    axs[2*q].text(x,obs['mu'][0][int(x)], r'$\mu$', color='white', size=20)
    sigs = [r'$+3\sigma$',r'$+2\sigma$',r'$+\sigma$',r'$\bar{x}_0$']
    x2 = 49
    ff = torch.Tensor([0,0,0,-1])
    for i in range(1,5):
        axs[2*q].text(x2,(obs['mu'][0]+quantiles[-i])[int(x2)], sigs[i-1], color='white', size=12, ha='center')  

    axs[2*q].set_ylabel(r'$\tilde{d}(f)$')

local_fix_plot(axs, tickdir='out')

plt.tight_layout()
plt.savefig(f'figs/{netid}/tmaps.png', dpi=700, bbox_inches = 'tight')

################### DO PVALUE PLOTS ###################

def pvalue_grid_eps(dat):
    eps_t_mean = np.mean(ts_bin_H0_epsilon, axis=0)
    eps_t_ref = ts_bin_H0_epsilon - eps_t_mean
    counts = np.sum(eps_t_ref >= dat[:, np.newaxis, :], axis=1)
    return (counts + 1) / (len(eps_t_ref) + 1)

def pvalue_grid_BCE(dat):
    BCE_t_mean = np.mean(ts_bin_H0_BCE, axis=0)
    BCE_t_ref = ts_bin_H0_BCE - BCE_t_mean
    counts = np.sum(BCE_t_ref >= dat[:, np.newaxis, :], axis=1)
    return (counts + 1) / (len(BCE_t_ref) + 1)

fig, ax1 = pf.create_plot()
ax1.set_xlabel(r'$f$')
ax2 = fig.add_axes((1.05, 0,0.1,1))

ax3 = fig.add_axes((0, 1.1,1,1))
plt.setp(ax3.get_xticklabels(), visible=False)
ax4 = fig.add_axes((1.05, 1.1,0.1,1))

ax3.set_title(f"Data: Training {mu_label} "+r'$\mu$'+f' On {eps_label} '+r'$\epsilon$')

axs = [ax1,ax2,ax3,ax4]

dat = [pvalue_grid_BCE(a), pvalue_grid_eps(s)]
lab =  [r'$\mathrm{log}_{10}($p$_{i, \mathrm{BCE}})$',r'$\mathrm{log}_{10}($p$_{i, \mathrm{SNR}})$']

labcolour = "#000000"


for q in range(2):
    mesh = axs[2*q].pcolormesh(position_grid.T, amplitude_grid.T, np.log10(dat[q]), cmap='magma', vmin=-8)
    fig.colorbar(mesh,cax=axs[2*q+1], shrink=0.8, label=lab[q])
    axs[2*q+1].set_ylim([-6.5,0])

    for j in range(2):
        axs[2*q].plot(chop_middle(positions)[j], chop_middle(obs['mu'][0])[j], color=labcolour, linewidth=3)
        for i in range(5):
            alp = .5+(i/8)
            axs[2*q].plot(chop_middle(positions)[j], chop_middle(obs['mu'][0]+quantiles[-i])[j], color=labcolour, alpha=alp)

    x = 47
    axs[2*q].text(x,obs['mu'][0][int(x)], r'$\mu$', color=labcolour, size=20)
    sigs = [r'$+3\sigma$',r'$+2\sigma$',r'$+\sigma$',r'$\bar{x}_0$']
    x2 = 49
    ff = torch.Tensor([0,0,0,-1])
    for i in range(1,5):
        axs[2*q].text(x2,(obs['mu'][0]+quantiles[-i])[int(x2)], sigs[i-1], color=labcolour, size=12, ha='center')  

    axs[2*q].set_ylabel(r'$\tilde{d}(f)$')

local_fix_plot(axs, tickdir='out')

plt.tight_layout()
plt.savefig(f'figs/{netid}/pmaps.png', dpi=700, bbox_inches = 'tight')


##### SLICE MAP #######

x_h0_all = np.load('../../data_bin/stats_ref/x_h0_all.npy')

kdebloc = np.load('../../data_bin/KDE_ref/KDE_archive.npz')
dat_h0 = kdebloc['dat_h0']+1e-21
dat_h1 = kdebloc['dat_h1']+1e-21

matrix = np.linspace(-30, 30, 1000)
mask = (matrix >= -2.0) & (matrix <= 4.5)
ti = -2 * (np.log(dat_h0)-np.log(dat_h1))
t_fn = interp1d(matrix[mask], ti[mask], bounds_error=False, fill_value=50.0)
t_samples = t_fn(x_h0_all) 

x_grid = np.linspace(-2, 8.0, 100)
actual_t_values = t_fn(x_grid)
num_extreme_vals_mask = t_samples > actual_t_values[:,
      np.newaxis]
num_extreme_vals = np.sum(num_extreme_vals_mask, axis=1)
p_values = num_extreme_vals / len(t_samples)
p_values[p_values == 0] = 1e-7

fig, ax1 = pf.create_plot(size=(4,1.5))
ax2 = fig.add_axes((0,1,1,1), sharex=ax1)
ax3 = fig.add_axes((0,2,1,1), sharex=ax1)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

grid = np.linspace(-30,30, 1000)
ti = -2 * (np.log(dat_h0)-np.log(dat_h1))

ax1.plot(grid,dat_h0, color='#BA1200', label=r'H$_0$', lw=3)
ax1.plot(grid,dat_h1, color='#39304A', label=r'H$_i$', lw=3)
ax1.legend()
ax1.set_xlim([-1.5,4.5])
ax1.set_xlabel(r'$\tilde{d}(f)$')
ax1.set_ylabel(r'$\mathbb{P}(\tilde{d}|...)$')

ax2.plot(grid,ti, lw=2,color='black')
ax2.set_ylabel(r'$-2\:\log\frac{p(\tilde{d}|H_0)}{p(\tilde{d}|H_i)}$')

ax3.plot(x_grid, p_values, lw=2, color='black', label='semi-analytical', zorder=10)
ax3.set_ylabel(r'p$_i$')
ax3.set_yscale('log')

# grid2 = dat[1]
# grid3 = dat[0]

labz = [r'BCE',r'eMu-s']
cz = [mycolors[0], '#ff004f']
for i in range(100):
    # randbin = np.random.randint(0,100)
    # randbin=0

    # randp = grid2[:,i]
    # randp_BCE = grid3[:,i]

    # ax3.plot(amplitudes, randp, lw=3, color='#ff004f', label= if i==0 else None, alpha=0.5)
    # ax3.plot(amplitudes, randp_BCE, lw=3, color=mycolors[0], label=r'BCE' if i==0 else None, alpha=0.5)

    muat = obs['mu'][0][i].numpy()
    amplitudes = np.linspace(-3, 10, 80)-muat

    for j in range(2):
        ax3.plot(amplitudes, dat[j][:,i], lw=3, color=cz[j], label=labz[j] if i==0 else None, alpha=0.5)


ax3.legend(fontsize=12)
ax3.set_ylim([3e-8,10])
pf.fix_plot([ax1,ax2,ax3])

plt.tight_layout()
plt.savefig(f'figs/{netid}/pdf2_log.png', dpi=700, bbox_inches = 'tight')
