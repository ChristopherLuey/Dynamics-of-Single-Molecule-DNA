import tifffile as tiff
import numpy as np
import os

# Function to generate a TIFF file with multiple particles at constant y_velocity
# Dim: (frames, height, width)
# Particles: [((y_0, x_0), frame_0, x_vel, particle_radius)]
def generate_tiff(dim, particles, output_file):
    data = np.zeros(dim)

    for frame in range(dim[0]):
        for particle in particles:
            initial_pos, entry_frame, velocity, particle_radius = particle
            if frame >= entry_frame:
                y_pos = initial_pos[0] + velocity * (frame - entry_frame)  # Move up by adding velocity to y
                x_pos = initial_pos[1]  # Keep x position constant
                if 0 <= y_pos < dim[1] and 0 <= x_pos < dim[2]:
                    for dy in range(-particle_radius, particle_radius + 1):
                        for dx in range(-particle_radius, particle_radius + 1):
                            if dx**2 + dy**2 <= particle_radius**2:
                                if 0 <= int(y_pos + dy) < dim[1] and 0 <= int(x_pos + dx) < dim[2]:
                                    data[frame, int(y_pos + dy), int(x_pos + dx)] = 1

    # Rotate the entire data array by 180 degrees
    data = np.rot90(data, 2, axes=(1, 2))

    # Flip the data along the x-axis
    data = np.flip(data, axis=2)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tiff.imwrite(output_file, data.astype(np.uint16))

if __name__ == "__main__":
    # Parameters
    frames = 1000
    height = 500
    width = 100
    # Particles: [((y_0, x_0), frame_0, x_vel, particle_radius)]
    particles = [
        ((0, 20), 0, 3, 2),
        ((0, 20), 100, 3, 5),
        ((0, 60), 20, 3, 3),
        ((0, 80), 400, 3.5, 3),
        ((0, 70), 400, 2.5, 1),



        # ((0, 50), 100, 3, 2),

    #    ((0,50), 20, 3, 8),
    #    ((0,20), 30, 2, 3),
    #     ((0,70), 80, 4, 2),

    ]

    # Output file
    output_file = 'syn_data/syn2.tif'

    # Generate TIFF
    generate_tiff((frames, height, width), particles, output_file)
