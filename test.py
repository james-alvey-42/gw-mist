import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

print("--- Step 1: Fetching and Preparing Real Detector Data ---")
# -- 1. Fetch the PSD for the LIGO Hanford detector around GW150914
# The psd() method of a TimeSeries will calculate the PSD.
# We can get the data from GWOSC using fetch_open_data.
event_time = event_gps('GW150914')
detector = 'H1'
start = event_time - 16
end = event_time + 16
print(f"Fetching {end - start} seconds of data for '{detector}' around event GW150914 (GPS time: {event_time})...")

# fetch the real data
data = TimeSeries.fetch_open_data(detector, start, end, sample_rate=4096)
print("Data fetched successfully.")

# calculate the PSD
print("Calculating Power Spectral Density (PSD) from the real data...")
psd = data.psd(fftlength=4)
print("PSD calculated.")

print("\n--- Step 2: Generating White Noise ---")
# -- 2. Generate white noise with the same properties
# We'll create 32 seconds of noise at the same sample rate (4096 Hz)
duration = 32
sample_rate = 4096
print(f"Generating {duration} seconds of Gaussian white noise using JAX...")
key = jax.random.PRNGKey(0)
noise_array = jnp.random.normal(key, shape=(int(duration * sample_rate),))
noise = TimeSeries(noise_array, sample_rate=sample_rate)
print("White noise generated.")

print("\n--- Step 3: Coloring the Noise ---")
# -- 3. Color the noise using the fetched PSD
# The inject() method handles the frequency-domain coloring
print("Coloring the white noise with the real data PSD...")
colored_noise = noise.inject(psd)
print("Noise colored successfully.")

print("\n--- Step 4: Plotting Results ---")
# -- 4. (Optional) Plot the results to verify
print("Generating plots to compare the generated noise with the real data...")
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Plot the time-domain colored noise
ax = axes[0]
ax.set_title(f"Generated '{detector}' Noise Strain")
ax.plot(colored_noise.times, colored_noise, linewidth=0.5)
ax.set_ylabel("Strain")
ax.set_xlabel("Time (s)")

# Plot the amplitude spectral density (ASD) to see the color
ax = axes[1]
ax.set_title("Amplitude Spectral Density (ASD)")
ax.plot(psd.frequencies, psd**0.5, label='Real Data ASD')
ax.plot(colored_noise.asd(fftlength=4).frequencies, colored_noise.asd(fftlength=4), label='Generated Noise ASD')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r'Strain [1/$\sqrt{\mathrm{Hz}}$]')
ax.set_xlabel("Frequency (Hz)")
ax.legend()
ax.grid(True, which='both')
ax.set_xlim(20, 2048)

plt.tight_layout()
print("Displaying plots. Close the plot window to exit the script.")
plt.show()
