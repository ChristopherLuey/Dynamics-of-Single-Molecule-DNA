import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import tifffile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Load TIFF data
tiff_data = tifffile.imread('1300kDa uncropped 22V for 30s.tif')

# Downsample the data for speed
downsample_factor = 1
tiff_data_downsampled = tiff_data[::downsample_factor, ::downsample_factor, ::downsample_factor]

# Get the dimensions of the downsampled TIFF data
num_frames, height, width = tiff_data_downsampled.shape
dim = (num_frames, height, width)

# Create a grid for the downsampled data
z = np.linspace(0, num_frames - 1, num_frames)
y = np.linspace(0, height - 1, height)
x = np.linspace(0, width - 1, width)
grid = (z, y, x)

# z: frames
# y: height
# x: width

# Neighboring Interpolator
interpolator = RegularGridInterpolator(grid, tiff_data_downsampled, method='linear', bounds_error=False, fill_value=0)

def get_cross_section(interpolator, point, angle_y, dim):
    xmax, ymax, zmax = dim[0], dim[1], dim[2]
    xo, yo, zo = point[0], point[1], point[2]
    unit_vector = np.array([np.cos(angle_y), 0, np.sin(angle_y)])

    vmax = ymax

    if zmax * np.tan(angle_y) > xmax - xo:
        umax = (xmax - xo) / np.sin(angle_y)
    else:
        umax = zmax / np.cos(angle_y)

    # Create grid on the plane
    u = np.linspace(0, umax - 1, int(umax))
    v = np.linspace(0, vmax - 1, int(vmax))
    uu, vv = np.meshgrid(u, v)
    
    # Coordinate transformation from plane space to original space
    xx = xo + uu * np.sin(angle_y)
    yy = yo + vv
    zz = zo + uu * np.cos(angle_y)
    
    # Flatten the grids for interpolation
    points = np.vstack((zz.flatten(), yy.flatten(), xx.flatten())).T

    # Interpolate the values on the plane
    values = interpolator(points).reshape((len(v), len(u)))
    
    return xx, yy, zz, values

# Initial values
z_frame_0 = z[0]  # Z coordinate of the first frame
_height = 0  # Initial height value
_width = 30
point = np.array([_width, _height, z_frame_0])
angle_y = np.pi / 10 

# Create a figure with subplots
fig = plt.figure(figsize=(12, 8))

# 3D view
ax1 = fig.add_subplot(121, projection='3d')
# Reduce the number of points for the 3D scatter plot
downsample_plot_factor = 50
X, Y, Z = np.meshgrid(x[::downsample_plot_factor], y[::downsample_plot_factor], z[::downsample_plot_factor], indexing='ij')
tiff_data_flat = tiff_data_downsampled[::downsample_plot_factor, ::downsample_plot_factor, ::downsample_plot_factor].flatten()
sc = ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=tiff_data_flat, cmap='hot', marker='o', alpha=0.1)
surf = None

# Cross-section view
ax2 = fig.add_subplot(122)
img = ax2.imshow(np.zeros((height, width)), cmap='hot')

# Slider for _height value
ax_angle_slider = plt.axes([0.25, 0.02, 0.65, 0.03])
angle_slider = Slider(ax_angle_slider, 'Angle', 0, np.pi, valinit=angle_y, valstep=0.1)

def update(val):
    angle_y = angle_slider.val
    print(f"Updating height to: {_height}")
    #point = np.array([_width, _height, z_frame_1])
    uu, vv, zz, values = get_cross_section(interpolator, point, angle_y, (width, height, num_frames))

    global surf
    if surf:
        surf.remove()
    surf = ax1.plot_surface(uu, vv, zz, color='red', alpha=0.5)


    img.set_data(values)
    ax2.set_title(f"Cross-Section at angle {angle_y}")
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    fig.canvas.draw_idle()

# Initialize the plot with the starting height
update(angle_y)

# Set initial axis labels
ax1.set_xlabel('Width (pixels)')
ax1.set_ylabel('Height (pixels)')
ax1.set_zlabel('Frames')
ax1.set_title("3D Visualization of TIFF Data with Cutting Plane")

angle_slider.on_changed(update)

plt.tight_layout()
plt.show()
