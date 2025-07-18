import numpy as np
from scipy.signal import welch


def Comb(
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
    Injects a LIGO comb signal into strain data by operating in the frequency domain.
    Each harmonic of the comb has a fixed amplitude, a random phase, and a frequency
    with lognormal jitter. The injection is performed on the Fourier transform of the
    data, and then converted back to a timeseries.
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
        The amplitude of each harmonic. Defaults to 1000.0.
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
    # 1. Setup time and frequency arrays for FFT
    num_samples = len(strain_data)
    if num_samples == 0:
        # Handle empty input gracefully
        return np.array([]), np.array([]), np.array([]), np.array([]), np.copy(strain_data)

    fft_freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)

    # 2. Generate comb parameters (frequencies, phases) in a vectorized manner
    base_frequencies = f0 + np.arange(nf) * df
    jitter_factors = np.random.lognormal(mean=0, sigma=jitter_sigma, size=nf)
    f_jittered = base_frequencies * jitter_factors
    phases = np.random.uniform(0, 2 * np.pi, size=nf)

    # 3. Create the comb signal in the frequency domain
    comb_fft = np.zeros_like(fft_freqs, dtype=np.complex128)

    # Find the frequency bin indices for each comb harmonic
    indices = np.searchsorted(fft_freqs, f_jittered)

    # Filter out harmonics that are outside the valid frequency range (e.g., > Nyquist)
    valid_mask = (indices > 0) & (indices < len(fft_freqs))
    indices = indices[valid_mask]
    phases = phases[valid_mask]
    
    # Proceed only if there are valid harmonics to add
    if indices.size > 0:
        # Calculate the complex amplitude for the FFT bins
        # For a real signal x(t) = A*cos(2*pi*f*t + phi), the rfft value is (A*N/2) * exp(j*phi)
        complex_amplitudes = (amplitude * num_samples / 2) * np.exp(1j * phases)

        # Add the complex amplitudes to the corresponding frequency bins.
        # Using np.add.at handles cases where multiple jittered frequencies fall into the same bin.
        np.add.at(comb_fft, indices, complex_amplitudes)

    # 4. Transform comb to time domain and calculate PSDs
    comb_timeseries = np.fft.irfft(comb_fft, n=num_samples)

    # 5. Calculate PSDs using scipy.signal.welch for the comb
    if nperseg is None:
        nperseg = int(sample_rate)

    window = ("tukey", tukey_alpha)

    welch_freqs, psd_comb = welch(
        comb_timeseries, fs=sample_rate, window=window, nperseg=nperseg, noverlap=nperseg // 2
    )

    # 6. Calculate the PSD of the combined signal directly from the FFT
    # This avoids an unnecessary inverse transform and preserves the original noise characteristics
    strain_fft = np.fft.rfft(strain_data)
    combined_fft = strain_fft + comb_fft
    
    # The PSD from an FFT is (2 / (fs * N)) * |X_k|^2 for a real signal
    # where N is the number of points in the window (nperseg)
    # For Welch's method, it's averaged over segments.
    # To simplify and align with the user's goal, we will return the FFT data
    # and let the user compute the PSD as needed, or we can provide a simplified PSD.
    # For now, let's stick to a direct calculation that matches the Welch output scale.
    
    # We will calculate the PSD of the strain data and add the comb's power.
    # This is a more direct way to get the combined PSD without IFFT->FFT.
    _, psd_strain = welch(
        strain_data, fs=sample_rate, window=window, nperseg=nperseg, noverlap=nperseg // 2
    )
    psd_combined = psd_strain + psd_comb

    # The combined timeseries is still useful for some applications
    combined_timeseries = np.fft.irfft(combined_fft, n=num_samples)
    
    extras = [combined_fft, strain_fft, comb_fft]
    return welch_freqs, psd_comb, psd_combined, comb_timeseries, combined_timeseries, extras


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

    # 3. Inject the comb using the new function
    # The 'extras' return value contains the raw FFTs, which we don't need for this plot.
    freqs, psd_comb, psd_combined, comb_ts, combined_ts, _ = Comb(
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
    fig.suptitle("LIGO Comb Injection Example (Frequency Domain)", fontsize=16)

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
    
    # For comparison, let's also plot the original noise PSD
    _, psd_noise = welch(strain_noise, fs=sample_rate, nperseg=int(sample_rate))
    ax2.loglog(freqs, psd_noise, label="Original Noise PSD", color='gray', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (strain$^2$/Hz)")
    ax2.legend()
    ax2.grid(True, which="both", ls=":")
    ax2.set_xlim(10, sample_rate / 2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()