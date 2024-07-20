import tifffile
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class TIFFWrapper:
    def __init__(self, file_name, downsample_factor=1):
        self.data = np.rot90(tifffile.imread(file_name), k=-1, axes=(1, 2))
        tifffile.imwrite(file_name.replace(".tif", "") + "_rotated.tif", self.data.astype(np.uint16))
        self.downsample_factor = downsample_factor
        self.num_frames, self.height, self.width = self.data.shape
        self.dim = (self.num_frames, self.height, self.width)


    def downsample(self, factor):
        self.data = self.data[::factor, ::factor, ::factor]
        self.downsample_factor = factor


    def mesh(self):
        self.z = np.linspace(0, self.num_frames - 1, self.num_frames)
        self.y = np.linspace(0, self.height - 1, self.height)
        self.x = np.linspace(0, self.width - 1, self.width)
        self.grid = (self.z, self.y, self.x)

        return self.grid
    

    def interpolator(self):
        self.interpolator = RegularGridInterpolator(self.mesh(), self.data, method='linear', bounds_error=False, fill_value=0) # type: ignore
        return self.interpolator
    

    def get_cross_section(self, point, angle_y, interpolator=None):
        xmax, ymax, zmax = self.dim[2], self.dim[1], self.dim[0]
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
        values = interpolator(points).reshape((len(v), len(u))) # type: ignore
        # print(values)
        
        return xx, yy, zz, values


    def graph(self, downsample_plot_factor=20):
        X, Y, Z = np.meshgrid(self.x[::downsample_plot_factor], self.y[::downsample_plot_factor], self.z[::downsample_plot_factor], indexing='ij')
        flatten = self.data[::downsample_plot_factor, ::downsample_plot_factor, ::downsample_plot_factor]

        return X.flatten(), Y.flatten(), Z.flatten(), flatten.flatten()
    

    def plot(self, point, angle_y, velocity, z_frame_0, _width, _height, interpolator, downsample_plot_factor_3d=20, downsample_plot_factor_filter=5, intensity_threshold=0.1):
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')

        # 3D scatter plot with downsampled data
        for factor, alpha in [(downsample_plot_factor_3d, 0.1), (downsample_plot_factor_filter, 0.3)]:
            X, Y, Z, flatten = self.graph(downsample_plot_factor=factor)
            if factor == downsample_plot_factor_filter:
                mask = flatten > intensity_threshold
                X, Y, Z, flatten = X[mask], Y[mask], Z[mask], flatten[mask]
            ax1.scatter(X, Y, Z, c=flatten, cmap='hot', marker='.', alpha=alpha)

        ax1.view_init(elev=-15, azim=-95)  # type: ignore

        # Set initial axis labels
        ax1.set_xlabel('Width (px)', fontsize=14)
        ax1.set_ylabel('Height (px)', fontsize=14)
        ax1.set_zlabel('Frames', fontsize=14)  # type: ignore
        ax1.set_title("3D Visualization of TIFF Data with Cutting Plane", fontsize=18)

        # 2D cross-section view
        ax2 = fig.add_subplot(gs[0, 1])
        img = ax2.imshow(np.zeros((self.height, self.width)), aspect='auto', cmap='hot')

        def update():
            nonlocal surf
            uu, vv, zz, values = self.get_cross_section(point, angle_y, interpolator)
            if surf:
                surf.remove()
            surf = ax1.plot_surface(uu, vv, zz, color='red', alpha=0.5)  # type: ignore
            img.set_data(values)
            if values.size > 0:
                img.set_clim(vmin=0, vmax=1)  # Update color scale limits

            ax2.set_title(f"Cross-Section at {velocity:.2f} px/frame (angle {angle_y:.2f})\nPlane at (Frame: {z_frame_0:.2f}, Width: {_width:.2f}, Height: {_height:.2f})", fontsize=18)
            ax2.set_xlabel('Spatial Temporal Dimension', fontsize=16)
            ax2.set_ylabel('Height (px)', fontsize=16)
            fig.canvas.draw_idle()

        surf = None
        update()

        # Manually set the position of the subplots
        pos1 = ax1.get_position()  # Get the original position of ax1
        pos2 = ax2.get_position()  # Get the original position of ax2

        # Adjust the position of ax1 and ax2
        ax1.set_position([pos1.x0-pos1.width/2, pos1.y0-pos1.height/2, pos1.width * 1.8, pos1.height * 1.8])  # Increase the size of ax1
        ax2.set_position([pos2.x0+pos2.width/11, pos2.y0-0.015, pos2.width * 1.2, pos2.height * 1.05])  # Decrease the size of ax2

        return fig, ax1, ax2
        