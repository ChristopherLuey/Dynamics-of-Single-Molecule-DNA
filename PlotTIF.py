import matplotlib.pyplot as plt
import numpy as np

array_2d = np.load("/Users/christopherluey/Desktop/DynamicsOfSingleMoleculeDNA/DynamicsOfSingleMoleculeDNA/syn_data/syn2.tif.npy")


fig, ax = plt.subplots(figsize=(12, 6))
cax = ax.imshow(array_2d, aspect='auto', cmap='hot')  # Use the 'hot' colormap
ax.set_xlabel('Width')
ax.set_ylabel('Height')
ax.set_title('Integrated Width Average for Particle Travelling at {} px/frame'.format(1))
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Average Intensity')

plt.show()
