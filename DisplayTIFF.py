import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import tifffile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Load TIFF data
tiff_data = tifffile.imread('data/1300kDa uncropped 22V for 30s.tif')

# Downsample the data for faster visualization
downsample_factor = 1
tiff_data_downsampled = tiff_data[::downsample_factor, ::downsample_factor, ::downsample_factor]

# Get the dimensions of the downsampled TIFF data
num_frames, height, width = tiff_data_downsampled.shape

# Corrected dim to include all three dimensions
dim = (num_frames, height, width)

print(dim)

# Create a grid for the downsampled data
z = np.linspace(0, num_frames - 1, num_frames)
y = np.linspace(0, height - 1, height)
x = np.linspace(0, width - 1, width)

# Create a figure with subplots
fig = plt.figure(figsize=(12, 8))

# 3D view
ax1 = fig.add_subplot(121, projection='3d')
# Reduce the number of points for the 3D scatter plot
downsample_plot_factor = 50
X, Y, Z = np.meshgrid(x[::downsample_plot_factor], y[::downsample_plot_factor], z[::downsample_plot_factor], indexing='ij')
tiff_data_flat = tiff_data_downsampled[::downsample_plot_factor, ::downsample_plot_factor, ::downsample_plot_factor].flatten()
sc = ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=tiff_data_flat, cmap='hot', marker='o', alpha=0.1)

plt.tight_layout()
plt.show()
