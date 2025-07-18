import numpy as np
from scipy.signal import welch
# gwpy is an optional dependency, required for this function.
# Please install it using: pip install gwpy
try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    print("Warning: 'gwpy' package not found. The Comb2 function will not be usable.")
    print("Please install it using: pip install gwpy")
    TimeSeries = None

def Comb2(
    sample_rate: float,
    f0: float,
    df: float,
    nf: int,
    amplitude_factor: float = 1e3,
    jitter_sigma: float = 1e-6,
    nperseg: int = None,
    tukey_alpha: float = 0.25,
    duration: int = 4,
    gps_start_time: int = 1126259462 - 2, # Default to 2s before GW150914
):
    """
    Generates a LIGO-like comb signal based on real noise data from GWOSC.

    This function fetches real strain data from the LIGO L1 detector, calculates its
    Power Spectral Density (PSD), and then creates a comb signal in the frequency
    domain on top of this noise background.

    Parameters:
    -----------
    sample_rate : float
        The sample rate for data fetching and processing in Hz.
    f0 : float
        The fundamental frequency of the comb in Hz.
    df : float
        The frequency spacing (separation) between comb harmonics in Hz.
    nf : int
        The number of harmonics (lines) in the comb.
    amplitude_factor : float, optional
        The factor by which the noise PSD is multiplied to create the comb peak amplitude.
        Defaults to 1e3.
    jitter_sigma : float, optional
        The standard deviation of the lognormal distribution for frequency jitter.
        The jitter is applied multiplicatively. Defaults to 1e-6.
    nperseg : int, optional
        Length of each segment for the Welch PSD calculation.
        If None, defaults to `sample_rate`.
    tukey_alpha : float, optional
        The shape parameter of the Tukey window used in the Welch PSD calculation.
        Defaults to 0.25.
    duration : int, optional
        The duration of the noise data to fetch from GWOSC in seconds. Defaults to 4.
    gps_start_time : int, optional
        The GPS start time for fetching the data. Defaults to 2 seconds before the
        GW150914 event.

    Returns:
    --------
    psd_noise : np.ndarray
        The Power Spectral Density of the fetched detector noise.
    psd_with_comb : np.ndarray
        The PSD of the noise with the comb signal injected.
    freqs : np.ndarray
        The frequency bins for the PSDs.
    comb_on_ones_timeseries : np.ndarray
        A time-domain signal representing the comb on a flat background (PSD=1).
    """
    if TimeSeries is None:
        raise ImportError("The 'gwpy' package is required to run this function.")

    # 1. Fetch real noise data from LIGO L1 using gwosc
    gps_end_time = gps_start_time + duration
    try:
        strain_data = TimeSeries.fetch_open_data(
            'L1', gps_start_time, gps_end_time, sample_rate=sample_rate,
        )
    except Exception as e:
        print(f"Could not fetch data from GWOSC. Please check your internet connection.")
        print(f"Error: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 2. Calculate the Power Spectral Density (PSD) of the noise
    if nperseg is None:
        nperseg = int(sample_rate)
    
    window = ("tukey", tukey_alpha)
    freqs, psd_noise = welch(
        strain_data.value, fs=sample_rate, window=window, nperseg=nperseg, noverlap=nperseg // 2
    )

    # 3. Create the comb in the frequency domain
    base_frequencies = f0 + np.arange(nf) * df
    jitter_factors = np.random.lognormal(mean=0, sigma=jitter_sigma, size=nf)
    f_jittered = base_frequencies * jitter_factors

    indices = np.searchsorted(freqs, f_jittered)
    valid_mask = (indices > 0) & (indices < len(freqs))
    indices = indices[valid_mask]

    psd_with_comb = np.copy(psd_noise)
    if indices.size > 0:
        comb_amplitudes = amplitude_factor * psd_noise[indices]
        np.add.at(psd_with_comb, indices, comb_amplitudes)

    # 4. Create a time-domain signal for the comb on a flat background
    psd_for_timeseries = np.ones_like(freqs)
    if indices.size > 0:
        comb_amplitudes_on_ones = amplitude_factor * psd_noise[indices]
        np.add.at(psd_for_timeseries, indices, comb_amplitudes_on_ones)

    num_samples_out = nperseg
    
    fft_amplitudes = np.sqrt(psd_for_timeseries * sample_rate * num_samples_out / 2.0)
    
    phases = np.random.uniform(0, 2 * np.pi, size=len(fft_amplitudes))
    complex_fft = fft_amplitudes * np.exp(1j * phases)
    
    complex_fft[0] = np.sqrt(psd_for_timeseries[0] * sample_rate * num_samples_out)
    
    if num_samples_out % 2 == 0:
        nyquist_index = len(fft_amplitudes) - 1
        complex_fft[nyquist_index] = np.sqrt(psd_for_timeseries[nyquist_index] * sample_rate * num_samples_out)

    comb_on_ones_timeseries = np.fft.irfft(complex_fft, n=num_samples_out)

    return psd_noise, psd_with_comb, freqs, comb_on_ones_timeseries

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Simulation Parameters ---
    sample_rate = 4096  # Hz
    f0 = 100  # Fundamental frequency in Hz
    df = 50  # Spacing in Hz
    nf = 20   # Number of harmonics

    # --- Generate the comb and noise PSDs ---
    try:
        psd_noise, psd_with_comb, freqs, comb_ts = Comb2(
            sample_rate=sample_rate,
            f0=f0,
            df=df,
            nf=nf,
            amplitude_factor=1e4, # Make peaks very prominent for plotting
            duration=4
        )
    except (ImportError, Exception) as e:
        print(f"Could not run example: {e}")
        exit()

    if psd_noise.size == 0:
        print("Failed to generate data, exiting example.")
        exit()

    # --- Plot the results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Comb Generation with Real LIGO Noise (Comb2)", fontsize=16)

    # Plot the PSDs
    ax1.set_title("Power Spectral Densities")
    ax1.loglog(freqs, psd_noise, label="Original L1 Noise PSD", alpha=0.7)
    ax1.loglog(freqs, psd_with_comb, label="Noise PSD with Comb", alpha=0.9)
    ax1.set_ylabel("PSD (strain$^2$/Hz)")
    ax1.legend()
    ax1.grid(True, which="both", ls=":")
    ax1.set_xlim(20, sample_rate / 2)

    # Plot the generated time-series
    # First, calculate the PSD of the generated timeseries to verify it matches
    nperseg = int(sample_rate)
    freqs_ts, psd_ts = welch(comb_ts, fs=sample_rate, nperseg=nperseg)
    
    ax2.set_title("Verification of Generated Time Series")
    ax2.loglog(freqs_ts, psd_ts, label="PSD of Generated Comb Timeseries")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (strain$^2$/Hz)")
    ax2.legend()
    ax2.grid(True, which="both", ls=":")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
