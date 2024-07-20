import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from TIFFWrapper import *
from SyntehticTIFFGen import *
import random
from PIL import Image
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({'font.size': 16})


output_dir = 'output_plots11'
os.makedirs(output_dir, exist_ok=True)

# Parameters
frames = 1000
height = 500
width = 100

# Particles: [((y_0, x_0), frame_0, x_vel, particle_radius)]
def generate_particles(num):
    particles = []
    with open(os.path.join(output_dir, 'particles30.txt'), 'w') as file:
        for i in range(num):
            n = ((0, random.randint(0,100)), random.randint(1, 999), random.randint(1,5), random.randint(2,5))
            file.write("%s\n" % str(n))
            particles.append(n)
    return particles

# Output file
tiff_name = os.path.join(output_dir,'syn30.tif')

# Generate TIFF
particles = generate_particles(30)
generate_tiff((frames, height, width), particles, tiff_name)

# Create TIF
tif = TIFFWrapper(tiff_name)
interpolator = tif.interpolator()

# def process_velocity(velocity):
for velocity in np.linspace(1, 5, 5):

    ################ Initial values #################
    #global tif
    z_frame = 0
    z_frame_0 = tif.z[z_frame]
    _height = 0
    _width = tif.width
    point = np.array([_width, _height, z_frame_0])  # point plane starts
    angle_y = np.arctan(velocity)

    frames = []  # Image

    #width_steps = tif.num_frames
    width_steps = tif.width
    step = tif.width / width_steps
    total_size = (tif.height, tif.num_frames + width_steps)
    counter = 0

    print("Velocity: {}\nAngle: {}".format(velocity, angle_y))

    uu, vv, zz, values = tif.get_cross_section(point, angle_y, interpolator)

    projected_plane = np.zeros(total_size)
    print(projected_plane.shape)

    row_averages = np.nan_to_num(np.nanmean(values, axis=1), nan=0.0)

    projected_plane[:,-counter-1] = row_averages

    while True:
        _width = _width - step
        if _width < 0.0:
            _width = 0.0
            z_frame += 1
            if z_frame >= tif.num_frames:
                break
            z_frame_0 = tif.z[z_frame]

        point = np.array([_width, _height, z_frame_0])
        #print(point)
        uu, vv, zz, values = tif.get_cross_section(point, angle_y, interpolator)
        row_averages = np.nan_to_num(np.nanmean(values, axis=1), nan=0.0)
        projected_plane[:, -counter - 2] = row_averages

        if counter % 10 == 0:
            # Create initial figure
            fig, ax1, ax2 = tif.plot(point, angle_y, velocity, z_frame_0, _width, _height, interpolator)
            fig.canvas.draw()  # Draw the figure so we can grab the RGBA buffer
            img1 = np.array(fig.canvas.renderer.buffer_rgba())  # type: ignore # Grab the RGBA buffer
            plt.close(fig)  # Close the figure to free up memory

            # Create additional figure
            fig2, ax3 = plt.subplots(figsize=(8, 8))
            cax = ax3.imshow(projected_plane, aspect='auto', cmap='hot', vmin=0, vmax=1)  # Use the 'hot' colormap with specified intensity scale
            ax3.set_xlabel('Spatial Temporal Dimension')
            ax3.set_ylabel('Height (px)')
            ax3.set_title(f'Integrated Width Average for Particle Travelling at \n{velocity:.2f} px/frame with {angle_y:.2f} angle')
            cbar = fig2.colorbar(cax, ax=ax3)
            cbar.set_label('Average Intensity')
            cax.set_clim(0, 1)  # Set the color limits for the ScalarMappable
            plt.tight_layout()
            fig2.canvas.draw()
            img2 = np.array(fig2.canvas.renderer.buffer_rgba())  # type: ignore
            plt.close(fig2)  # Close the figure to free up memory

            # Combine the two images side by side
            combined_width = img1.shape[1] + img2.shape[1]
            combined_height = max(img1.shape[0], img2.shape[0])
            combined_img = np.zeros((combined_height, combined_width, 4), dtype=np.uint8)

            # Place img1 and img2 side by side in combined_img
            combined_img[:img1.shape[0], :img1.shape[1], :] = img1
            combined_img[:img2.shape[0], img1.shape[1]:, :] = img2

            # Convert the combined image to a PIL Image and append to frames
            frames.append(Image.fromarray(combined_img))
            plt.show()

        counter += 1
        #print(counter)

    frames[0].save(os.path.join(output_dir, f'gif_velocity_{velocity:.2f}.gif'), save_all=True, append_images=frames[1:], duration=30, loop=0)

    np.save(os.path.join(output_dir, f'data_velocity_{velocity:.2f}.npy'), projected_plane)
    print("Velocity: {}\nAngle: {}".format(velocity, angle_y))

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(projected_plane, aspect='auto', cmap='hot')  # Use the 'hot' colormap
    ax.set_xlabel('Spatial Temporal Dimension')
    ax.set_ylabel('Height')
    ax.set_title(f'Integrated Width Average for Particle Travelling at \n{velocity:.2f} px/frame with {angle_y:.2f} angle')
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Average Intensity')
    plot_filename = os.path.join(output_dir, f'plot_velocity_{velocity:.2f}.png')
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to free memory
    print(f"Plot saved to {plot_filename}")

# if __name__ == "__main__":
#     velocities = np.linspace(1, 5, 5)
#     with Pool() as pool:
#         pool.map(process_velocity, velocities)