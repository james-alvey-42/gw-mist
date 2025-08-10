import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings

# Suppress Matplotlib warnings about tight_layout
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

# --- Placeholder functions and data (to make the example runnable) ---

def get_snr2(b):
    """Placeholder for your SNR calculation."""
    return torch.rand_like(b['x']) * 10

def pvalue_grid_eps(s):
    """Placeholder for your p-value calculation."""
    return 10**(-s / 2.0)

class PlottingFunctions:
    """Placeholder for your plotting setup."""
    def create_plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(right=0.85)
        return fig, ax

pf = PlottingFunctions()

def chop_middle(data):
    """Placeholder for your chopping function."""
    size = data.shape[0]
    split_point = max(1, size // 2)
    return data[:split_point], data[split_point:]

# --- Main Script ---

# 1. Setup initial parameters and data
Nbins = 100
labcolour = "#000000"
positions = torch.arange(0, Nbins, 1, dtype=torch.float32)

# Gaussian parameters
gauss_mean = 50.0
gauss_std = 20.0

# Sigma bands and labels
quantiles = torch.tensor([3.0, 2.0, 1.0, 0.0]) # Corresponds to +3, +2, +1 sigma and the mean
sigs_labels = [r'$+3\sigma$', r'$+2\sigma$', r'$+\sigma$', r'$\bar{x}_0$']

# 2. Set up the static plot elements
fig, ax1 = pf.create_plot()
ax1.set_xlabel(r'$f$')
ax1.set_ylabel(r'$\tilde{d}(f)$')
cax = fig.add_axes([0.88, 0.12, 0.04, 0.76])

amplitudes_bg = torch.linspace(-3, 20, 80)
position_grid, amplitude_grid = torch.meshgrid(positions, amplitudes_bg, indexing='ij')
s = get_snr2({'x': amplitude_grid.T})
dat = pvalue_grid_eps(s)

mesh = ax1.pcolormesh(position_grid, amplitude_grid, np.log10(dat.numpy()), cmap='magma', vmin=-8, vmax=0)
fig.colorbar(mesh, cax=cax, label=r'$\mathrm{log}_{10}($p$_{i, \mathrm{SNR}})$')
cax.set_ylim([-6.5, 0])
ax1.set_ylim(torch.min(amplitudes_bg).item(), torch.max(amplitudes_bg).item())
ax1.text(gauss_mean - 3, 5.5, r'$\mu$', color=labcolour, size=20) # Static mu label

# 3. Initialize the animated artists (lines and text)
x_chopped1, x_chopped2 = chop_middle(positions)

# Main line for the gaussian
main_line1, = ax1.plot([], [], color=labcolour, linewidth=3)
main_line2, = ax1.plot([], [], color=labcolour, linewidth=3)

# Sigma band lines and text labels
sigma_lines1, sigma_lines2, sigma_texts = [], [], []
for i in range(len(quantiles)):
    l1, = ax1.plot([], [], color=labcolour, alpha=0.3 + (i / 8.0))
    l2, = ax1.plot([], [], color=labcolour, alpha=0.3 + (i / 8.0))
    sigma_lines1.append(l1)
    sigma_lines2.append(l2)
    txt = ax1.text(0, 0, sigs_labels[i], color=labcolour, size=12, ha='center', visible=False)
    sigma_texts.append(txt)

# 4. Define the animation logic
num_frames = 60
# Create a smooth sinusoidal oscillation for the amplitude, ranging from 1 to 5
# Midpoint = (5+1)/2 = 3. Fluctuation = (5-1)/2 = 2.
anim_amplitudes = 3.0 + 2.0 * torch.sin(torch.linspace(0, 2 * np.pi, num_frames))

def gaussian(x, amp, mean, std):
    return amp * torch.exp(-((x - mean)**2) / (2 * std**2))

def init():
    """Initializes the animation artists."""
    main_line1.set_data([], [])
    main_line2.set_data([], [])
    for i in range(len(quantiles)):
        sigma_lines1[i].set_data([], [])
        sigma_lines2[i].set_data([], [])
        sigma_texts[i].set_visible(False)
    return [main_line1, main_line2] + sigma_lines1 + sigma_lines2 + sigma_texts

def update(frame):
    """Update function for the animation."""
    current_amp = anim_amplitudes[frame]
    
    # Calculate the main gaussian curve for the current amplitude
    y_main = gaussian(positions, current_amp, gauss_mean, gauss_std)
    y_main_c1, y_main_c2 = chop_middle(y_main)
    main_line1.set_data(x_chopped1.numpy(), y_main_c1.numpy())
    main_line2.set_data(x_chopped2.numpy(), y_main_c2.numpy())

    # Update sigma bands and their labels
    label_pos_x = gauss_mean + 2 # x-position for labels
    for i in range(len(quantiles)):
        # The sigma bands are offsets from the main gaussian
        y_sigma = y_main + quantiles[i]
        y_sigma_c1, y_sigma_c2 = chop_middle(y_sigma)
        sigma_lines1[i].set_data(x_chopped1.numpy(), y_sigma_c1.numpy())
        sigma_lines2[i].set_data(x_chopped2.numpy(), y_sigma_c2.numpy())
        
        # Update text position and make it visible
        sigma_texts[i].set_position((label_pos_x, y_sigma[int(label_pos_x)]))
        sigma_texts[i].set_visible(True)

    return [main_line1, main_line2] + sigma_lines1 + sigma_lines2 + sigma_texts

# 5. Create and save the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    init_func=init,
    blit=True,
    interval=50
)

output_filename = 'gaussian_animation_final.gif'
print(f"Saving animation to {output_filename}...")
ani.save(output_filename, writer='pillow', fps=15)
print("Done.")

plt.close(fig)