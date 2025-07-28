import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_sigma_background(ax, quantile_values):
    """
    Plots a colored background on a matplotlib axis to represent 1, 2, and
    3-sigma confidence intervals.

    The bands are drawn behind other plot elements.

    Args:
        ax (matplotlib.axes.Axes): The axis object to draw on.
        quantile_values (list or np.ndarray): A sorted array of 7 floats
            representing the data values at the quantiles:
            [-3sig, -2sig, -1sig, mean, +1sig, +2sig, +3sig].
    """
    if len(quantile_values) != 7:
        raise ValueError("quantile_values must be an array of 7 elements.")

    # Define nice, semi-transparent colors for the bands
    colors = {
        "1_sigma": "#a1c9f4",  # Pastel Blue
        "2_sigma": "#ffb482",  # Pastel Orange
        "3_sigma": "#ff9f9b",  # Pastel Red
    }

    # Unpack quantile values for clarity
    q_neg3, q_neg2, q_neg1, _, q_pos1, q_pos2, q_pos3 = quantile_values

    # Plot 3-sigma bands (from -3 to -2 and +2 to +3)
    # These are drawn first to be in the back
    ax.axvspan(q_neg3, q_neg2, alpha=0.6, color=colors["3_sigma"], lw=0, zorder=-3)
    ax.axvspan(q_pos2, q_pos3, alpha=0.6, color=colors["3_sigma"], lw=0, zorder=-3)

    # Plot 2-sigma bands (from -2 to -1 and +1 to +2)
    ax.axvspan(q_neg2, q_neg1, alpha=0.6, color=colors["2_sigma"], lw=0, zorder=-2)
    ax.axvspan(q_pos1, q_pos2, alpha=0.6, color=colors["2_sigma"], lw=0, zorder=-2)

    # Plot 1-sigma band (from -1 to +1)
    # This is drawn last to be visually on top of the other bands
    ax.axvspan(q_neg1, q_pos1, alpha=0.6, color=colors["1_sigma"], lw=0, zorder=-1, label='1-3 $\sigma$ intervals')
    
    return True

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define some sample quantile data.
    # In a real scenario, you would calculate these from your data.
    # Format: [-3sig, -2sig, -1sig, mean, +1sig, +2sig, +3sig]
    quantiles = [-2.5, -1.8, -0.9, 0.1, 1.1, 2.2, 3.5]

    # 2. Set up the plot
    # Using a professional plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['figure.dpi'] = 150
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Plot the sigma background FIRST, so it's in the background
    print("Plotting sigma confidence bands...")
    plot_sigma_background(ax, quantiles)

    # 4. Plot your actual data on TOP of the background.
    # As an example, we'll plot a simple Gaussian-like curve.
    print("Plotting example data...")
    x_data = np.linspace(-4, 4, 200)
    # A curve centered around the mean of our quantiles
    y_data = np.exp(-0.5 * (x_data - quantiles[3])**2) 
    ax.plot(x_data, y_data, color='black', linewidth=2, label='Example Data')

    # 5. Add a vertical line for the mean
    ax.axvline(quantiles[3], color='black', linestyle='--', linewidth=1.5, label=f'Mean ({quantiles[3]})')

    # 6. Finalize the plot with labels, title, and legend
    ax.set_title("Plot with Colored Sigma Intervals", fontsize=16)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Probability Density (or other metric)", fontsize=12)
    
    # Place legend nicely
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    # Set axis limits to nicely frame the data and bands
    ax.set_xlim(quantiles[0] - 0.5, quantiles[-1] + 0.5)
    ax.set_ylim(0, 1.2)

    # 7. Save the figure to a file
    output_filename = 'sigma_background_example.png'
    print(f"Saving example plot to '{output_filename}'...")
    try:
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved plot to '{output_filename}'.")
    except IOError as e:
        print(f"Error: Could not save plot. Reason: {e}")
    
    # To display the plot in an interactive session:
    # plt.show()
