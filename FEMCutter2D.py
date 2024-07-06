from TIFFWrapper import *
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from matplotlib.widgets import Slider


# tiff = TIFFWrapper('data/1300kDa uncropped 22V for 30s.tif')
tiff = TIFFWrapper('syn_data/syn_larger_particles_radial.tif')
interpolator = tiff.interpolator()

################ Initial values #################
z_frame_0 = tiff.z[0]
_height = 0 
_width = 0
point = np.array([_width, _height, z_frame_0]) # point plane starts
angle_y = np.pi / 4  # angle of cut

################ Plotting ####################
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121, projection='3d')

# Reduce the number of points for the 3D scatter plot
X,Y,Z, flatten = tiff.graph()
sc = ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=flatten, cmap='hot', marker='o', alpha=0.1)
surf = None

# Cross-section view
ax2 = fig.add_subplot(122)
img = ax2.imshow(np.zeros((tiff.height, tiff.width)), aspect='auto', cmap='hot')

# Slider for _height value
ax_angle_slider = plt.axes([0.25, 0.02, 0.65, 0.03])
angle_slider = Slider(ax_angle_slider, 'Angle', 0, np.pi, valinit=angle_y, valstep=0.1)

ax_width_slider = plt.axes([0.25, 0.07, 0.65, 0.08])
width_slider = Slider(ax_width_slider, 'Width', 0, 300, valinit=_width, valstep=0.1)

def update(val):
    angle_y = angle_slider.val
    _width = width_slider.val
    point = np.array([_width, _height, z_frame_0])
    uu, vv, zz, values = tiff.get_cross_section(point, angle_y, interpolator)

    global surf
    if surf:
        surf.remove()
    surf = ax1.plot_surface(uu, vv, zz, color='red', alpha=0.5)
    img = ax2.imshow(values, aspect="auto", cmap="hot")
    img.set_clim(vmin=values.min(), vmax=values.max())  # Update color scale limits
    plt.draw()
    ax2.set_title(f"Cross-Section at angle {angle_y}")
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    fig.canvas.draw_idle()

update(angle_y)

# Set initial axis labels
ax1.set_xlabel('Width (pixels)')
ax1.set_ylabel('Height (pixels)')
ax1.set_zlabel('Frames')
ax1.set_title("3D Visualization of TIFF Data with Cutting Plane")

angle_slider.on_changed(update)
width_slider.on_changed(update)


plt.tight_layout()
plt.show()

