import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import tifffile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Load TIFF data
tiff_data = tifffile.imread('1300kDa uncropped 22V for 30s.tif')

# Downsample the data for faster visualization
downsample_factor = 4
tiff_data_downsampled = tiff_data[::downsample_factor, ::downsample_factor, ::downsample_factor]

# Get the dimensions of the downsampled TIFF data
num_frames, height, width = tiff_data_downsampled.shape
print(num_frames, height, width)

# Create a grid for the downsampled data
z = np.linspace(0, num_frames - 1, num_frames)
y = np.linspace(0, height - 1, height)
x = np.linspace(0, width - 1, width)
grid = (z, y, x)

# Create the interpolator function for the downsampled data
interpolator = RegularGridInterpolator(grid, tiff_data_downsampled, method='linear', bounds_error=False, fill_value=0)

def get_cross_section(interpolator, point, vector, grid_height, grid_width):
    # Normalize the vector
    vector = vector / np.linalg.norm(vector)
    
    # Create a grid on the plane
    u = np.linspace(0, grid_height - 1, grid_height)
    v = np.linspace(0, grid_width - 1, grid_width)
    uu, vv = np.meshgrid(u, v)
    
    # Define the normal vector to the plane
    normal_vector = np.cross(vector, [1, 0, 0]) if vector[1] == 0 else np.cross(vector, [0, 1, 0])
    
    # Avoid division by zero
    normal_vector[normal_vector == 0] = 1e-10
    
    # Plane equation: ax + by + cz = d
    a, b, c = normal_vector
    d = np.dot(normal_vector, point)
    
    # Calculate the z values on the plane for each (u, v)
    zz = (-d - a * uu - b * vv) / c
    
    # Flatten the grids for interpolation
    points = np.vstack((zz.flatten(), uu.flatten(), vv.flatten())).T

    # Clip points to be within the grid bounds
    points_clipped = np.clip(points, [0, 0, 0], [num_frames - 1, height - 1, width - 1])

    # Interpolate the values on the plane
    values = interpolator(points_clipped).reshape((grid_height, grid_width))
    
    return uu, vv, zz, values

# Initial values
z_frame_1 = z[1]  # Z coordinate of the second frame
_height = 30  # Initial height value
_width = 30
point = np.array([z_frame_1, _height, _width])
vector = np.array([1, 1/5, 1/2])  # Example vector defining the plane's orientation

# Create a figure with subplots
fig = plt.figure(figsize=(12, 8))

# 3D view
ax1 = fig.add_subplot(121, projection='3d')
# Reduce the number of points for the 3D scatter plot
downsample_plot_factor = 10
X, Y, Z = np.meshgrid(x[::downsample_plot_factor], y[::downsample_plot_factor], z[::downsample_plot_factor], indexing='ij')
tiff_data_flat = tiff_data_downsampled[::downsample_plot_factor, ::downsample_plot_factor, ::downsample_plot_factor].flatten()
sc = ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=tiff_data_flat, cmap='gray', marker='o', alpha=0.1)
surf = None

# Cross-section view
ax2 = fig.add_subplot(122)
img = ax2.imshow(np.zeros((height, width)), cmap='gray')

# Slider for _height value
ax_height_slider = plt.axes([0.25, 0.02, 0.65, 0.03])
height_slider = Slider(ax_height_slider, 'Height', 0, height - 1, valinit=_height, valstep=1)

def update(val):
    _height = height_slider.val
    print(f"Updating height to: {_height}")
    point = np.array([z_frame_1, _height, _width])
    uu, vv, zz, values = get_cross_section(interpolator, point, vector, height, width)

    if values.max() != 0:
        values = (values - values.min()) / (values.max() - values.min())
    
    global surf
    if surf:
        surf.remove()
    surf = ax1.plot_surface(vv, uu, zz, color='red', alpha=0.5)
    img.set_data(values)
    ax2.set_title(f"Cross-Section at height {_height}")
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    fig.canvas.draw_idle()

# Initialize the plot with the starting height
update(_height)

# Set initial axis labels
ax1.set_xlabel('Width (pixels)')
ax1.set_ylabel('Height (pixels)')
ax1.set_zlabel('Frames')
ax1.set_title("3D Visualization of TIFF Data with Cutting Plane")

height_slider.on_changed(update)

plt.tight_layout()
plt.show()
