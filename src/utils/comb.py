import numpy as np
from scipy.signal import welch

def inject_ligo_comb(
    strain_data: np.ndarray,
    sample_rate: float,
    f0: float,
    df: float,
    nf: int,
    amplitude: float = 1000.0,
    jitter_sigma: float = 1e-6,
    nperseg: int = None,
    tukey_alpha: float = 0.25,
):
    """
    Injects a LIGO comb signal into a strain data timeseries and computes PSDs.

    Each harmonic of the comb has a fixed amplitude and a frequency with
    lognormal jitter.

    Parameters:
    -----------
    strain_data : np.ndarray
        Time-dependent LIGO strain wave function.
    sample_rate : float
        The sample rate of the strain data in Hz.
    f0 : float
        The fundamental frequency of the comb in Hz.
    df : float
        The frequency spacing (separation) between comb harmonics in Hz.
    nf : int
        The number of harmonics (lines) in the comb.
    amplitude : float, optional
        The amplitude of each harmonic. Defaults to 1000.0 as requested.
    jitter_sigma : float, optional
        The standard deviation of the lognormal distribution for frequency jitter.
        The jitter is applied multiplicatively. Defaults to 1e-6.
    nperseg : int, optional
        Length of each segment for the Welch PSD calculation.
        If None, defaults to the value of `sample_rate`.
    tukey_alpha : float, optional
        The shape parameter of the Tukey window used in the Welch PSD calculation.
        Defaults to 0.25.

    Returns:
    --------
    freqs : np.ndarray
        The frequency bins for the PSDs.
    psd_comb : np.ndarray
        The Power Spectral Density of the generated comb signal alone.
    psd_combined : np.ndarray
        The Power Spectral Density of the strain data with the comb injected.
    comb_timeseries : np.ndarray
        The time series of the generated comb signal.
    combined_timeseries : np.ndarray
        The time series of the strain data with the comb injected.
    """
    # 1. Generate time array from input strain data
    num_samples = len(strain_data)
    time = np.arange(num_samples) / sample_rate

    # 2. Generate the comb signal
    comb_timeseries = np.zeros_like(strain_data)
    base_frequencies = f0 + np.arange(nf) * df

    for f_base in base_frequencies:
        # Apply multiplicative lognormal jitter to the base frequency.
        # For lognormal(mean=0, sigma), the median of the distribution is exp(0) = 1.
        jitter_factor = np.random.lognormal(mean=0, sigma=jitter_sigma)
        f_jittered = f_base * jitter_factor
        comb_timeseries += amplitude * np.sin(2 * np.pi * f_jittered * time)

    # 3. Create the combined signal by adding the comb to the original data
    combined_timeseries = strain_data + comb_timeseries

    # 4. Calculate PSDs using scipy.signal.welch
    if nperseg is None:
        nperseg = int(sample_rate)  # Default to 1-second segments

    window = ("tukey", tukey_alpha)

    # PSD of the comb signal alone
    freqs, psd_comb = welch(
        comb_timeseries,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    # PSD of the combined signal
    # The frequency bins (`freqs`) will be identical to the above call
    _, psd_combined = welch(
        combined_timeseries,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    return freqs, psd_comb, psd_combined, comb_timeseries, combined_timeseries


# if __name__ == "__main__":
    # This block provides an example of how to use the function.
    # It will only run when the script is executed directly.
    import matplotlib.pyplot as plt

    # --- Simulation Parameters ---
    sample_rate = 4096  # Hz
    duration = 10  # seconds
    num_samples = duration * sample_rate

    # 1. Create a dummy strain signal (e.g., Gaussian noise)
    np.random.seed(42)
    # A realistic noise level for LIGO data
    strain_noise = np.random.normal(0, 1e-23, num_samples)

    # 2. Define comb parameters
    f0 = 50  # Fundamental frequency in Hz
    df = 25  # Spacing in Hz
    nf = 8   # Number of harmonics

    # 3. Inject the comb using the function
    # Note: Using a more realistic amplitude for the example plot to make it visible.
    freqs, psd_comb, psd_combined, comb_ts, combined_ts = inject_ligo_comb(
        strain_data=strain_noise,
        sample_rate=sample_rate,
        f0=f0,
        df=df,
        nf=nf,
        amplitude=5e-22, # Using a smaller amplitude for a clearer example plot
        jitter_sigma=1e-7
    )

    # 4. Plot the results for verification
    time_axis = np.arange(num_samples) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("LIGO Comb Injection Example", fontsize=16)

    # Plot a small segment of the time series to see the injection
    ax1.set_title("Time Series (first 0.25 seconds)")
    ax1.plot(time_axis, combined_ts, label="Combined (Noise + Comb)", alpha=0.9)
    ax1.plot(time_axis, comb_ts, label="Comb Signal Only", alpha=0.8, linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Simulated Strain")
    ax1.legend()
    ax1.grid(True, which="both", ls=":")
    ax1.set_xlim(0, 0.25)

    # Plot the Power Spectral Densities
    ax2.set_title("Power Spectral Density (Welch Method)")
    ax2.loglog(freqs, psd_combined, label="Combined (Noise + Comb) PSD")
    ax2.loglog(freqs, psd_comb, label="Comb Signal PSD", alpha=0.8, linestyle="--")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (strain$^2$/Hz)")
    ax2.legend()
    ax2.grid(True, which="both", ls=":")
    ax2.set_xlim(10, sample_rate / 2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
