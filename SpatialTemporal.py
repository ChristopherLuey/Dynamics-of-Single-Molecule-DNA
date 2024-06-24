import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# Read the multi-frame TIFF file
file_path = 'filtered/20/1300kDa_uncropped_22V_for_30s_temporal_filtered.tif'
tiff_data = tiff.imread(file_path)
print(f'TIFF data shape: {tiff_data.shape}')

# Data shape is (frames, height, width)
num_frames, height, width = tiff_data.shape

# Calculate the width average for each frame
width_averages = tiff_data.mean(axis=2)

# Create a grid to display the width averages
fig, ax = plt.subplots(figsize=(12, 6))
cax = ax.imshow(width_averages, aspect='auto', cmap='hot')  # Use the 'hot' colormap
ax.set_xlabel('Width')
ax.set_ylabel('Time (s)')
ax.set_title('Width Average of Each Frame')

# Set the y-ticks and labels to show time (in seconds)
frame_interval = 25
time_per_frame = 50 / 1000  # each frame represents 50 ms, converted to seconds
y_ticks = np.arange(0, num_frames, frame_interval)
y_labels = (y_ticks * time_per_frame)[::-1]  # Reverse the labels
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Add a color bar to show the scale of the averages
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Average Intensity')

plt.show()
