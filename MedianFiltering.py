import numpy as np
import tifffile as tiff
from scipy.ndimage import median_filter

# Read the multi-frame TIFF file
file_path = 'data/1300kDa uncropped 22V for 30s.tif'
tiff_data = tiff.imread(file_path)
print(f'TIFF data shape: {tiff_data.shape}')

# Data shape is (frames, height, width)
num_frames, height, width = tiff_data.shape

# Apply median filtering in the temporal dimension
tiff_data_transposed = np.moveaxis(tiff_data, 0, -1)  # shape becomes (height, width, frames)
filtered_data_transposed = np.empty_like(tiff_data_transposed)

for i in range(height):
    for j in range(width):
        # Apply median filter along the time axis (which is now the last axis)
        filtered_data_transposed[i, j] = median_filter(tiff_data_transposed[i, j], size=20)

# Move the frames axis back to the first position
filtered_data = np.moveaxis(filtered_data_transposed, -1, 0)  # shape becomes (frames, height, width)

# Save the filtered data as a new TIFF file
filtered_file_path = 'filtered/20/1300kDa_uncropped_22V_for_30s_temporal_filtered.tif'
tiff.imwrite(filtered_file_path, filtered_data)

print(f'Filtered TIFF file saved to: {filtered_file_path}')


# Perimeter is noisy, can't track edges, 
# Fit intensities to computing area moments to compute centroid, choose elipse that has most intensity power, extract major and minor axis
# does this assume that the shape deforms into an ellipse


# FEA: analytical function represent data, 
