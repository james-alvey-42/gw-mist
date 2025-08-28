import gwosc
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import numpy as np

event_time = event_gps('GW150914')
detector = 'H1'
psd_window = 100
start = event_time - 200
end = event_time - 100
data = TimeSeries.fetch_open_data(detector, event_time-20, event_time+20, sample_rate=4096, verbose=True)
psd = data.psd(fftlength=8)

# --- New code to select a random chunk ---

# Define the chunk duration
chunk_duration = 8

# Calculate the latest possible start time for an 8s chunk
latest_start_offset = data.duration.value - chunk_duration

# Select a random start time offset
random_start_offset = np.random.uniform(0, latest_start_offset)

# Calculate the absolute start and end times for the crop
random_chunk_start = data.t0.value + random_start_offset
random_chunk_end = random_chunk_start + chunk_duration

# Extract the 8-second chunk
random_chunk = data.crop(random_chunk_start, random_chunk_end)

print(f"Selected a random {len(random_chunk)} sample chunk.")
print(f"Chunk start time: {random_chunk.t0}")
print(f"Chunk duration: {random_chunk.duration}")
