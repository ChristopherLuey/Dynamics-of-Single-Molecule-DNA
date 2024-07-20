from TIFFWrapper import *
import numpy as np
import matplotlib.pyplot as plt
import tifffile

#tiff = TIFFWrapper('data/1300kDa uncropped 22V for 30s.tif')
tiff = TIFFWrapper('output_plots11/syn30.tif')
interpolator = tiff.interpolator()

################ Initial values #################
z_frame_0 = 286
_height = 0 
_width = 0
point = np.array([_width, _height, z_frame_0])  # point plane starts
velocity = 2  # px/frame
angle_y = np.arctan(velocity)

################ Plotting ####################
fig, ax1, ax2 = tiff.plot(point, angle_y, velocity, z_frame_0, _width, _height, interpolator)
plt.show()