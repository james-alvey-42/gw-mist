import torch
torch.set_float32_matmul_precision('medium')
import numpy as np
import argparse
import pytorch_lightning as pl
from tqdm import tqdm
import plotfancy as pf
import matplotlib.pyplot as plt
pf.housestyle_rcparams()

import os, sys
sys.path.append('../../mist-base/GW')
sys.path.append('../../mist-base/')
sys.path.append('../../mist-base/utils')
sys.path.append('../../')

from src.utils.generators import Simulator_Additive
from simulators.utils import *
from utils.data import OnTheFlyDataModule
from utils.module import CustomLossModule_withBounds


mycolors = ['#570f6d', "#9e8f92", '#f98e08']
folly = '#ff004f'

########### ARGPARSE AND GLOBALS ##############

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
#locks
parser.add_argument('--unlock_mu', action='store_false', help='Unlock Stoch Mu')
parser.add_argument('--unlock_amp', action='store_false', help='Unlock Stoch Amp')
parser.add_argument('--unlock_sigma', action='store_false', help='Unlock Stoch Mu')


args = parser.parse_args()

glob_sigma = args.sigma
glob_bkg = args.nobkg
glob_pve_bounds = args.pvebounds
glob_det = args.det
glob_mode = args.mode

Nsims = args.nsims
Nbins = args.nbins
train_bounds = args.bounds

glob_locks = [args.unlock_mu, args.unlock_amp, args.unlock_sigma] # mu amp sigma -> TRUE is locked, FALSE unlocks

# glob_sigma = 1
# glob_bkg = True
# glob_pve_bounds = False
# glob_det = 'stoch'


# Nsims = 100_000
# Nbins = 100s
# train_bounds = 5

simulator = Simulator_Additive(Nbins=Nbins, sigma=glob_sigma, bounds=train_bounds, 
                               fraction=0.2, bkg=glob_bkg, dtype=torch.float64, 
                               mode=glob_mode, pve_bounds=glob_pve_bounds, bump=glob_det,
                               lock_mu=glob_locks[0],lock_amp=glob_locks[1], lock_sigma=glob_locks[2])

samples = simulator.sample(Nsims=Nsims)  
obs = simulator.sample(1)

p_marker = 'p' if glob_pve_bounds == True else 'n'
b_marker = 'b' if glob_bkg == True else 'q'
d_marker = 'd' if glob_det == 'det' else 's_'
s_number = ''.join([str(i + 1) 
                    for i, lock in enumerate(glob_locks) 
                    if (i <= 2 and not lock)])+'_'

netid = 'eMu-d_'+p_marker+b_marker+d_marker+s_number+str(train_bounds)
print(f'Running a {netid} Network on 4000 epochs')

if not os.path.isdir('figs/'+netid):
    os.makedirs('figs/'+netid)
test = simulator.sample(1)
quantiles_long = np.array([7.1508466e-04, 7.9613253e-03, 5.1986761e-02,
       2.1462703e-01, 5.8794379e-01, 1.1776060e+00,
       1.9190179e+00, 2.7507384e+00, 3.6350725e+00,
       4.5491748e+00, 5.4850187e+00], dtype=np.float32)

quantiles = np.array([5.1986761e-02,
       2.1462703e-01, 5.8794379e-01, 1.1776060e+00,
       1.9190179e+00, 2.7507384e+00, 3.6350725e+00], dtype=np.float32)

pf.housestyle_rcparams()
fig, ax1 = pf.create_plot()

plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = fig.add_axes((0,-.3,1,0.3), sharex=ax1)
ax3 = fig.add_axes((1,-.3,0.2,0.3), sharey=ax2)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
ax4 = fig.add_axes((0,1,1,0.3), sharex=ax1)
plt.setp(ax4.get_xticklabels(), visible=False)

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

for i in range(100):
    ax4.plot(simulator.sample(1)['mu'][0], lw=0.5, color='black', alpha=0.5)

ax4.set_yticks([])

pf.fix_plot([ax1,ax2, ax3,ax4])
plt.tight_layout()

plt.tight_layout()
plt.savefig(f'figs/{netid}/distribution.png', bbox_inches='tight', dpi=400)

from models.online_norm import OnlineStandardizingLayer
from models.resnet_1d import ResidualNet

class Network_epsilon(torch.nn.Module):
    def __init__(self, nbins):
        super().__init__()
        
        self.nbins = nbins

        self.logvariance_mu = torch.nn.Parameter(torch.ones(self.nbins)*5)
        self.logvariance_epsilon = torch.nn.Parameter(torch.ones(self.nbins)*5)

        self.net = ResidualNet(1, 1, hidden_features=128, num_blocks=2, kernel_size=1, padding=0) 

        self.mu_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.nbins, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.nbins)
        )
    
    def get_x_like(self,x): # returns a dict
        x_shape = x.shape #[nsims, nbins]
        return simulator.sample(Nsims=x.shape[0])

    def mu(self, x):
        x = self.mu_predictor(x.unsqueeze(1)).squeeze(1)
        return x
                
    def epsilon(self, x):
        resd = x - self.mu(x)
        out = self.net(resd.unsqueeze(1)).squeeze(1) # x-net
        return out
    
    def snr(self, x):
        return self.epsilon(x) / self.logvariance_epsilon.exp().sqrt()  # [B, N_bins]
    
    def bounds(self):
        return self.logvariance_epsilon.detach().exp().sqrt().mean(-1) * 5
        
    def forward(self, x):
        
        otf_sample = self.get_x_like(x['x0'])
        x0_block  = otf_sample['x0'].cuda()
        mu_block =  otf_sample['mu'].cuda()
        ni = otf_sample['ni'].cuda()
        
        ###########################################
        epsilon_sim =  (2 * self.bounds() * torch.rand(x['x'].shape, 
                                                           device= x['x'].device, 
                                                           dtype= x['x'].dtype) - self.bounds()) * ni
        ###########################################

        data = x0_block+epsilon_sim
        
        # net evaluation_m
        net_mu = self.mu(data)
        error_mu = (net_mu-mu_block)**2
        l_mu = error_mu / (self.logvariance_mu.exp() + 1e-10) + self.logvariance_mu
        l_mu_return = l_mu.sum() * 0.5

        # net evaluation_e
        net_epsilon = self.epsilon(data)
        mask = ( ni != 0 )  
        squared_error_e = (net_epsilon - epsilon_sim)**2                                                  # [B, N_bins]
        l_e = squared_error_e / (self.logvariance_epsilon.exp() + 1e-10) + self.logvariance_epsilon                     # [B, N_bins]
        l_e_return = (l_e * mask.float()).sum() * 0.5
        
        # combine
        return l_mu_return+l_e_return


print('Training...')

batch_size = 124
samples = simulator.sample(Nsims=Nsims)  
dm = OnTheFlyDataModule(simulator, Nsims_per_epoch=400*batch_size, batch_size=batch_size, num_workers=31)

network_epsilon = Network_epsilon(nbins=Nbins)
model = CustomLossModule_withBounds(network_epsilon, learning_rate=3e-3)
trainer = pl.Trainer(
    accelerator="gpu", 
    max_epochs=4000, 
    precision=64,
    # fast_dev_run=True
)
trainer.fit(model, dm)
network_epsilon.cuda().eval();

torch.save(network_epsilon, f'networks/network_{netid}_complex')
torch.save(model, f'networks/model_{netid}_complex')

print('Training Complete! Model Saved')

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
sample = simulator.sample(1)
ni = sample['ni']

fig, axs = plt.subplots(4, 5, figsize=(20, (4+1)*3), sharex=True)
axs = axs.flatten()
for i_b, b in enumerate(bounds_history):
    if i_b < len(axs):
        axs[i_b].set_title(r"$b$ = {:.2f}".format(b))
        for j in range(10):
            sample = simulator.sample(1)
            ni = sample['ni']
            # axs[i_b].plot(sample['mu'][0].cpu(), c='k', ls='--')
            epsilon_sim =  (2 * b * torch.rand(sample['xi'].shape, device= sample['xi'].device, dtype= sample['xi'].dtype) - b) * ni
            data =  sample['x0'] + epsilon_sim * ni
            axs[i_b].plot(data[0].cpu(), c='C0', alpha=0.4)
# plt.tight_layout()
# plt.savefig(f'figs/{netid}/history.png', dpi=300)

print('History Plots Completed')

ox = simulator.sample(124)
testmu = ox['x0']
netout = network_epsilon.mu(ox['xi'].cuda())
nn = netout.detach().cpu().numpy()

fig, ax1 = pf.create_plot()
ax2 = fig.add_axes((0,1,1,0.4), sharex=ax1)
ax3 = fig.add_axes((0,-.3,1,0.3), sharex=ax1)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)

for i in range(len(nn)):
    ax1.plot(ox['mu'][i]+2, color='grey', alpha=0.2, label=('true' if i==0 else None))
    ax1.plot(nn[i], color='red', alpha=0.4, label=(f'eMu-d' if i==0 else None))

    resd = -ox['mu'][i]+nn[i]
    ax3.scatter(np.arange(100), resd, alpha=.5, color='black', s=2)

ax1.legend()
ax2.plot(network_epsilon.logvariance_epsilon.cpu().detach().exp().sqrt().numpy(), color='black', label=r'$\varepsilon$')
ax2.plot(network_epsilon.logvariance_mu.cpu().detach().exp().sqrt().numpy(), color='#dd78ae', label='$\mu$')
ax2.legend()
ax3.set_xlabel(r'$f$')
ax1.set_ylabel(r'$\tilde{d}(f)$ [range]')
ax2.set_ylabel(r'$\sigma_{\mathrm{NN}}$')
ax3.set_ylabel('resd')

ax1.set_yticks([])
pf.fix_plot([ax1,ax2,ax3])

plt.tight_layout()
plt.savefig(f'figs/{netid}/network_outputs.png', dpi=400, bbox_inches='tight')

print('Output Plot Complete')

network_epsilon.cuda()
N_mc = 2e6

obs = simulator.sample(1)
ni = torch.eye(Nbins, dtype=obs['xi'].dtype)
variance = 1 / get_sigma_epsilon_inv2(ni)

batch_size = 2048*2
N_batch = int(N_mc / batch_size)

ts_bin_H0_epsilon = []
for _ in tqdm(range(N_batch)):
    mc_samples = simulator.sample(batch_size)
    ts_batch =  (network_epsilon.snr(mc_samples['x0'].cuda())**2).detach().cpu().numpy()
    ts_bin_H0_epsilon.append(ts_batch)
    
ts_bin_H0_epsilon = np.concatenate(ts_bin_H0_epsilon)

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

def chop_middle(array, remove=5, linemode=True):
    if len(array)%2==0:
        mid_u = int(len(array)/4)
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
    snr2_nn = network_epsilon.snr(target.cuda()).detach().cpu().numpy()**2 
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
###### SET UP GRID ######
positions = torch.arange(0, Nbins, 1).to(dtype=simulator.dtype)
amplitudes = torch.linspace(-3, 10, 80).to(dtype=simulator.dtype)

position_grid, amplitude_grid = torch.meshgrid(positions,amplitudes)
b = {'x':amplitude_grid.T}

###### DO BCE STATS ######
a = b['x']
s = get_snr2(b)
dat = [a,s]


fig, ax1 = pf.create_plot()
ax1.set_xlabel(r'$f$')
ax2 = fig.add_axes((1.05, 0,0.1,1))

# ax3 = fig.add_axes((0, 1.1,1,1))
# plt.setp(ax3.get_xticklabels(), visible=False)
# ax4 = fig.add_axes((1.05, 1.1,0.1,1))

axs = [ax1,ax2]


dat = [pvalue_grid_eps(s)]
lab =  [r'$\mathrm{log}_{10}($p$_{i, \mathrm{SNR}})$']

labcolour = "#000000"

obs = simulator.sample(100)


for q in range(1):
    mesh = axs[2*q].pcolormesh(position_grid.T, amplitude_grid.T, np.log10(dat[q]), cmap='magma', vmin=-8)
    fig.colorbar(mesh,cax=axs[2*q+1], shrink=0.8, label=lab[q])
    axs[2*q+1].set_ylim([-6.5,0])

    for j in range(2):
        axs[2*q].plot(chop_middle(positions)[j], chop_middle(obs['mu'][0])[j], color=labcolour, linewidth=3)
        for i in range(5):
            alp = .5+(i/8)
            axs[2*q].plot(chop_middle(positions)[j], chop_middle(obs['mu'][0]+quantiles[-i])[j], color=labcolour, alpha=alp)

    x = 25
    axs[2*q].text(x,obs['mu'][0][int(x)], r'$\mu$', color=labcolour, size=20)
    sigs = [r'$+3\sigma$',r'$+2\sigma$',r'$+\sigma$',r'$\bar{x}_0$']
    x2 = 25
    ff = torch.Tensor([0,0,0,-1])
    for i in range(4):
        axs[2*q].text(x2,(obs['mu'][0]+quantiles[-i])[int(x2)], sigs[i-1], color=labcolour, size=12, ha='center')  

    axs[2*q].set_ylabel(r'$\tilde{d}(f)$')

local_fix_plot(axs, tickdir='out')

plt.tight_layout()
plt.savefig(f'figs/{netid}/pmaps.png', dpi=700, bbox_inches = 'tight')

print('Hetmap Plot Complete')


import matplotlib.animation as animation
# --- Main Script ---

# Create a gaussian-like shape for the base signal obs['mu']
positions = torch.arange(0, Nbins, 1, dtype=simulator.dtype)
mean_gaussian = 8 * torch.exp(-((positions - 50)**2) / (2 * 15**2))
obs = {'mu': mean_gaussian.unsqueeze(0)}

gauss_mean = 50.0
gauss_std = 100/24
gauss_amp = 3

sigs_labels = [r'$+3\sigma$', r'$+2\sigma$', r'$+\sigma$', r'$\bar{x}_0$']

# Define the range of the animation based on some quantiles
# This will be the max deviation from the mean signal
max_deviation = 5.0

# 2. Set up the static plot elements that will not change between frames
dim1 = 6
dim2 = (3/4)*dim1
fig, dummy = pf.create_plot(size=(dim1,dim2))
dummy.set_axis_off() 
ax1 = fig.add_axes((1/(1.35*dim1),1/dim2,4/dim1,3/dim2))
ax1.set_xlabel(r'$f$')
ax1.set_ylabel(r'$\tilde{d}(f)$')
# Explicitly define the colorbar axis to avoid layout issues
ax2 = fig.add_axes((4/dim1+0.15,1/dim2,0.05,3/dim2))

# Calculate and draw the pcolormesh background ONCE
positions = torch.arange(0, Nbins, 1).to(dtype=simulator.dtype)
amplitudes = torch.linspace(-3, 10, 80).to(dtype=simulator.dtype)
position_grid, amplitude_grid = torch.meshgrid(positions,amplitudes)
b = {'x':amplitude_grid.T}
s = get_snr2(b)
dat = pvalue_grid_eps(s)

mesh = ax1.pcolormesh(position_grid.T, amplitude_grid.T, np.log10(dat), cmap='magma', vmin=-8, vmax=0)
fig.colorbar(mesh, cax=ax2, label=r'$\mathrm{log}_{10}($p$_{i, \mathrm{SNR}})$')
ax2.set_ylim([-6.5, 0])

# Add static text annotations
# ax1.text(47, obs['mu'][0][47] + 0.5, r'$\mu$', color=labcolour, size=20)
ax1.set_ylim(torch.min(amplitudes).item(), torch.max(amplitudes).item())

# ax1.text(gauss_mean - 3, 5.5, r'$\mu$', color=labcolour, size=20) # Static mu label

# 3. Initialize the animated artists (lines and text)
x_chopped1, x_chopped2, _ = chop_middle(positions)

# Main line for the gaussian
main_line1, = ax1.plot([], [], color=labcolour, linewidth=3)
main_line2, = ax1.plot([], [], color=labcolour, linewidth=3)

# Sigma band lines and text labels
sigma_lines1, sigma_lines2, sigma_texts = [], [], []

for i in range(4):
    l1, = ax1.plot([], [], color=labcolour, alpha=0.2 + (i / 8.0))
    l2, = ax1.plot([], [], color=labcolour, alpha=0.2 + (i / 8.0))
    sigma_lines1.append(l1)
    sigma_lines2.append(l2)
    txt = ax1.text(0, 0, sigs_labels[i], color=labcolour, size=15, ha='center', visible=False)
    sigma_texts.append(txt)

mu_text = ax1.text(0,0, r'$\mu$', color=labcolour, size=15, ha='center', visible=False)

# 4. Define the animation logic
num_frames = 60
# Create a smooth sinusoidal oscillation for the amplitude, ranging from 1 to 5
anim_amplitudes = 3.5 + (1/2) * torch.sin(torch.linspace(0, 2 * np.pi, num_frames))
anim_means = 60 + 10 * torch.sin(torch.linspace(0, 2 * np.pi, num_frames))
anim_stds = (100/24+4) + (8/2)*torch.sin(torch.linspace(0, 2 * np.pi, num_frames))

def gaussian(x, amp, mean, std):
    return amp * torch.exp(-((x - mean)**2) / (2 * std**2))

def init():
    """Initializes the animation artists."""
    main_line1.set_data([], [])
    main_line2.set_data([], [])
    for i in range(4):
        sigma_lines1[i].set_data([], [])
        sigma_lines2[i].set_data([], [])
        sigma_texts[i].set_visible(False)
    mu_text.set_visible(False)
    return [main_line1, main_line2] + sigma_lines1 + sigma_lines2 + sigma_texts + [mu_text]

def update(frame):
    """Update function for the animation."""
    current_mean = anim_means[frame] if not glob_locks[0] else gauss_mean
    current_amp = anim_amplitudes[frame] if not glob_locks[1] else gauss_amp
    current_std = anim_stds[frame] if not glob_locks[2] else gauss_std
    
    # Calculate the main gaussian curve for the current amplitude
    y_main = gaussian(positions, current_amp, current_mean, current_std)
    y_main_c1, y_main_c2, _ = chop_middle(y_main)
    main_line1.set_data(x_chopped1.numpy(), y_main_c1.numpy())
    main_line2.set_data(x_chopped2.numpy(), y_main_c2.numpy())

    # Update sigma bands and their labels
    label_pos_x = 25 # x-position for labels
    for i in range(4):
        # The sigma bands are offsets from the main gaussian
        y_sigma = y_main + quantiles[-(i+1)]
        y_sigma_c1, y_sigma_c2, _ = chop_middle(y_sigma)
        sigma_lines1[i].set_data(x_chopped1.numpy(), y_sigma_c1.numpy())
        sigma_lines2[i].set_data(x_chopped2.numpy(), y_sigma_c2.numpy())
        
        # Update text position and make it visible
        sigma_texts[i].set_position((label_pos_x, y_sigma[int(label_pos_x)]))
        sigma_texts[i].set_visible(True)

    mu_text.set_position((label_pos_x, y_main[25]))
    mu_text.set_visible(True)

    return [main_line1, main_line2] + sigma_lines1 + sigma_lines2 + sigma_texts + [mu_text]



local_fix_plot([ax1,ax2], tickdir='out')

# 5. Create and save the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=(num_frames),
    init_func=init,
    blit=True,  # This is the key performance optimization
    interval=50 # Delay between frames in milliseconds
)

# Save the final GIF. You may need to install pillow: pip install pillow
output_filename = f'figs/{netid}/gaussian_range_animation.gif'
print(f"Saving animation to {output_filename}...")
ani.save(output_filename, writer='pillow', fps=15, dpi=700, savefig_kwargs={'pad_inches':0.5})
print("Done.")

plt.close(fig) # Prevent the static plot from displaying
obbs = simulator.sample(2048)
amp = obbs['theta'][:,1]
mu = obbs['mu']
xi = obbs['xi']
x0 = obbs['x0']
nn = network_epsilon.mu(x0.cuda()).cpu().detach().numpy()
# err = np.max(nn-mu.numpy(), axis=1)
err = (nn-mu.numpy())[:,50]
# plt.scatter(err, amp)
fig, ax = pf.create_plot()
ax.scatter(amp, err, alpha=0.2, color=folly, s=5)
ax.set_xlabel(r'$\mu$-Amplitude [$\tilde{d}(f)$]')
ax.set_ylabel(r'NN Error [bin 50]')
pf.fix_plot([ax])

plt.tight_layout()
plt.savefig(f'figs/{netid}/wrongness.png', dpi=700, bbox_inches = 'tight')
print('Finished!')
