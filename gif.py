import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import scipy.stats
import imageio
import os
import concurrent.futures

# --- Placeholder Implementations ---
# Please replace these with your actual classes, functions, and variables.

class DummyPlotFunctions:
    def housestyle_rcparams(self):
        # Placeholder for your rcParams setup
        pass
    def create_plot(self, size):
        fig, ax = plt.subplots(figsize=size)
        return fig, ax
    def fix_plot(self, axes):
        # Placeholder for your plot fixing function
        pass

pf = DummyPlotFunctions()

# Dummy variables
mycolors = ['#0077BB', '#33BBEE', '#EE7733', '#CC3311']
mygold = '#FFDD44'
Nbins = 100
N_mc = 1000
DOF = 1.0
quantiles_long = np.random.rand(5, Nbins)
ts_sum_H0_BCE = np.random.randn(500)
ts_sum_H0_epsilon = np.random.randn(500)


class Simulator_Additive:
    """
    Placeholder Simulator class.
    """
    def __init__(self, Nbins, sigma, bkg, bounds, fraction, dtype, mode):
        self.Nbins = Nbins
        self.dtype = dtype

    def sample(self, n_samples):
        # Returns a dictionary with dummy data of the expected shape
        return {
            'xi': torch.randn(n_samples, self.Nbins, dtype=self.dtype),
            'ni': torch.randn(n_samples, self.Nbins, dtype=self.dtype),
            'x0': torch.randn(n_samples, self.Nbins, dtype=self.dtype),
            'mu': torch.randn(n_samples, self.Nbins, dtype=self.dtype)
        }

def analyse_obs_BCE(obs):
    """
    Placeholder analysis function.
    """
    # Returns a tuple of dummy tensors
    n = obs['xi'].shape[1]
    return (torch.randn(n), torch.randn(n), torch.rand(n), torch.rand(n),
            torch.rand(1), torch.rand(1), torch.rand(1))

def analyse_obs_epsilon(obs):
    """
    Placeholder analysis function.
    """
    # Returns a tuple of dummy tensors and floats
    n = obs['xi'].shape[1]
    return (torch.randn(n), torch.randn(n), torch.rand(n), torch.randn(n),
            torch.randn(n), torch.rand(n), torch.rand(n), torch.rand(1),
            torch.rand(1), torch.rand(1))

# --- End of Placeholder Implementations ---


def plot_together_new(
    obs,
    ts_bin_obs, ts_bin_analytical, p_nn_BCE, p_analytical_BCE, p_sum_nn_BCE, p_sum_analytical_BCE, p_glob_all_BCE,
    snr2_nn, snr2_analytical, p_nn_epsilon, p_analytical_epsilon, p_sum_nn_epsilon, p_sum_analytical_epsilon, p_glob_all_epsilon
    ):

    pf.housestyle_rcparams()
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

    global xi
    xi = obs['xi'][0]
    ni = obs['ni'][0]
    dist = obs['xi'][0] - obs['x0'][0]
    grid = torch.linspace(0, Nbins, Nbins)
    ax1.plot(grid, obs['mu'][0], color='k', label=r"$\mu_{\mathrm{sim}}$")

    for i in range(1,6):
        ax1.fill_between(grid, 0, quantiles_long[-i],  color='#b0b0b0', alpha=0.1)
        ax_hist.fill_between(grid, 0, quantiles_long[-i],  color='#b0b0b0', alpha=0.1)


    ax1.scatter(grid, xi, c='k', marker='x', s=6)
    ax1.plot(grid, obs['mu'][0]+dist, color="#ff7b5a", label=r"$\mu_{\mathrm{dist}}$")
    ax1.set_ylabel(r"$x_\mathrm{obs}$", labelpad=1.5)
    ax1.legend(bbox_to_anchor=(0.05, 1), fontsize=13, loc='upper left', labelspacing=0.1)
    ax1.set_ylim(0, 6.5)
    ax1.set_title("Data")

    ####### NEW HISTOGRAM #########
    ax_hist.hist(xi, orientation='horizontal', color='white',
                 edgecolor='black', density=True, bins=np.linspace(0,6,17))
    ax_hist.set_xlim([0,0.7])


    ax2.semilogy(grid, p_nn_BCE, c=mycolors[0], label=r"$\mathrm{BCE}$")
    ax2.semilogy(grid, p_nn_epsilon, c='#ff004f', label=r"$\mathrm{SNR}$")
    ax2.semilogy(grid, p_analytical_BCE, c='k', ls='dotted', label=r"Analytical")
    ax2.set_ylabel(r"$\mathrm{p}_\mathrm{obs}$", labelpad=1.5)
    ax2.set_ylim(1/(N_mc*5), 1)
    ax2.set_title("Anomaly detection")
    ax2.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax2.yaxis.set_minor_locator(plt.NullLocator())
    ax2.grid(True, axis='y', which='major')
    ax2.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    ax2.set_yticklabels([r"$10^{-1}$", None , r"$10^{-3}$", None, r"$10^{-5}$", None])

    ax3.plot(grid, ts_bin_obs-ts_bin_obs.min(), c=mycolors[0], label=r"$\mathrm{BCE}$")
    ax3.plot(grid, snr2_nn-snr2_nn.min(), c='#ff004f', label=r"$\mathrm{SNR}$")
    ax3.plot(grid, ts_bin_analytical - ts_bin_analytical.min(), c='k', ls='dotted',label=r"Analytical")
    ax3.set_ylabel(r"$t_\mathrm{obs}$")
    ax3.legend(loc='upper right', fontsize=13)
    ax3.set_ylim(-.1, None)
    ax3.set_title(r"Localized $t_i$")

    distortion_locations = grid[ni != 0]
    for plot_ax in [ax1, ax2, ax3]:
        for loc in distortion_locations:
            plot_ax.axvline(loc, color='black', lw=1, zorder=-10)

    ax4.set_yscale('log')
    ax4.set_ylim(1/(N_mc*5), 1)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["BCE", "SNR"])
    ax4.set_xlim(-0.5, 2 - 0.5)
    ax4.set_ylabel(r"$\mathrm{p}_\mathrm{sum}$")
    ax4.fill_between([-0.2, 0.2], 1, p_sum_nn_BCE, facecolor=mycolors[0], edgecolor='black')
    ax4.fill_between([0.8, 1.2], 1, p_sum_nn_epsilon, facecolor='#ff004f', edgecolor='black')
    ax4.set_title(r"Model validation")
    ax4.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax4.yaxis.set_minor_locator(plt.NullLocator())
    ax4.grid(True, axis='y', which='major')
    ax4.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    ax4.set_yticklabels([r"$10^{-1}$", None , r"$10^{-3}$", None, r"$10^{-5}$", None])

    bins = np.linspace(ts_sum_H0_BCE.min(), ts_sum_H0_BCE.max(), 100)
    ax6.hist(ts_sum_H0_BCE, bins=bins, density=True, color=mycolors[0], alpha=0.5, label='BCE')
    ax6.hist(ts_sum_H0_BCE, bins=bins, density=True, color=mycolors[0], histtype='step')
    ax6.hist(ts_sum_H0_epsilon, bins=bins, density=True, color='#ff004f', alpha=0.5, label='SNR')
    ax6.hist(ts_sum_H0_epsilon, bins=bins, density=True, color='#ff004f', histtype='step')
    ax6.plot(bins, scipy.stats.chi2.pdf(bins, df=DOF, loc=-DOF), c='k', linestyle = 'dotted', label='Analytical')
    ax6.set_xlabel(r"$t_\mathrm{sum}$ $\mid H_0$")
    ax6.set_title(r"$\chi^2$")

    mantissa, exp = ('%.2e' % p_glob_all_BCE).split('e')
    exp = int(exp)
    textstr = r'$\mathrm{p}_{\mathrm{glob, BCE}}= %s \times 10^{%d}
 % (mantissa, exp)
    ax_text.text(0.5, 0.7, textstr, transform=ax_text.transAxes, fontsize=15,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(facecolor=mygold, edgecolor='none'))
    mantissa, exp = ('%.2e' % p_glob_all_epsilon).split('e')
    exp = int(exp)
    textstr = r'$\mathrm{p}_{\mathrm{glob, SNR}}= %s \times 10^{%d}
 % (mantissa, exp)
    ax_text.text(0.5, 0.4, textstr, transform=ax_text.transAxes, fontsize=15,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(facecolor=mygold, edgecolor='none'))
    ax_text.axis('off')
    ax_text.set_title("Global p-value")
    pf.fix_plot([ax1,ax2,ax3,ax4,ax6, ax_hist])

    plt.tight_layout()
    return fig

def generate_frame(frame_index, temp_dir):
    """
    Generates a single frame for the GIF.
    This function is designed to be called in a separate process.
    """
    print(f"Generating frame {frame_index+1}...")
    # --- Replace with your actual data simulation and analysis ---
    simulator1 = Simulator_Additive(Nbins=Nbins, sigma=1, bkg=True, bounds=5, fraction=0.05, dtype=torch.float32, mode='complex')
    obs = simulator1.sample(1)

    ts_bin_obs, ts_bin_analytical, p_nn_BCE, p_analytical_BCE, p_sum_nn_BCE, p_sum_analytical_BCE, p_glob_all_BCE = analyse_obs_BCE(obs)
    epsilon_nn, epsilon_analytical, variance_nn, snr2_nn, snr2_analytical, p_nn_epsilon, p_analytical_epsilon, p_sum_nn_epsilon, p_sum_analytical_epsilon, p_glob_all_epsilon = analyse_obs_epsilon(obs)
    # ---

    fig = plot_together_new(
        obs,
        ts_bin_obs, ts_bin_analytical, p_nn_BCE, p_analytical_BCE, p_sum_nn_BCE, p_sum_analytical_BCE, p_glob_all_BCE,
        snr2_nn, snr2_analytical, p_nn_epsilon, p_analytical_epsilon, p_sum_nn_epsilon, p_sum_analytical_epsilon, p_glob_all_epsilon
    )

    frame_path = os.path.join(temp_dir, f'frame_{frame_index:02d}.png')
    fig.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    return frame_path

def main():
    """
    Generates 20 plots in parallel and combines them into a GIF.
    """
    from itertools import repeat
    num_frames = 20
    gif_filename = 'animated_comparison.gif'
    temp_dir = 'temp_frames'

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Use ProcessPoolExecutor to run frame generation in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # The map function is often simpler and can be more robust.
        # It applies `generate_frame` to each item in `range(num_frames)`.
        results = executor.map(generate_frame, range(num_frames), repeat(temp_dir))
        frame_filenames = list(results)

    # Sort filenames to ensure correct order in GIF
    frame_filenames.sort()

    # Create the GIF
    print(f"Creating GIF: {gif_filename}")
    with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
        for filename in frame_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up temporary files
    print("Cleaning up temporary frames...")
    for filename in frame_filenames:
        os.remove(filename)
    os.rmdir(temp_dir)

    print(f"Successfully created {gif_filename}")

if __name__ == '__main__':
    main()
