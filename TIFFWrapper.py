import tifffile
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class TIFFWrapper:
    def __init__(self, file_name, downsample_factor=1):
        self.data = tifffile.imread(file_name)
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
        self.interpolator = RegularGridInterpolator(self.mesh(), self.data, method='linear', bounds_error=False, fill_value=0)
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
        values = interpolator(points).reshape((len(v), len(u)))
        print(values)
        
        return xx, yy, zz, values


    def graph(self, downsample_plot_factor=70):
        X, Y, Z = np.meshgrid(self.x[::downsample_plot_factor], self.y[::downsample_plot_factor], self.z[::downsample_plot_factor], indexing='ij')
        flatten = self.data[::downsample_plot_factor, ::downsample_plot_factor, ::downsample_plot_factor].flatten()

        return X, Y, Z, flatten
        

    

