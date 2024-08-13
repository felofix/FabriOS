import numpy as np
import pywavefront
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

class FindMyFeeders:
    def __init__(self, resolution, file=None, up_vec=np.array([0, 0, 1])):
        self.file = file
        self.up_vec = up_vec
        self.resolution = resolution

        # Check if file is provided in correct format
        if len(file[0]) > 1:
            self.vertices = file[0]
            self.faces = file[1]
            self.normals = []
        else:
            self.obj_to_point()  # Load data from .obj file

        self.find_dimensions()  # Calculate dimensions of the shape

    def obj_to_point(self):
        # Load and convert .obj file to vertices and faces
        file = self.file
        scene = pywavefront.Wavefront(file, create_materials=True, collect_faces=True)
        points = np.array(scene.vertices)
        self.vertices = points
        self.faces = scene.mesh_list[0].faces

    def show_shape(self):
        # Visualize the shape as a scatter plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.grid_points[:, 0], self.grid_points[:, 1], self.grid_points[:, 2], color="black", label='shape', alpha=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def show_heatmap(self):
        # Visualize heatmap and shape points
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(self.grid_points[:, 0], self.grid_points[:, 1], self.grid_points[:, 2], c=self.heatmap, cmap='viridis')
        ax.scatter3D(self.feeder_points[:, 0], self.feeder_points[:, 1], self.feeder_points[:, 2], color="black", label='shape', alpha=0.3)
        fig.colorbar(p)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def find_dimensions(self):
        # Find and store the min and max values for x, y, z dimensions
        min_x = np.min(self.vertices[:, 0])
        max_x = np.max(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        max_y = np.max(self.vertices[:, 1])
        min_z = np.min(self.vertices[:, 2])
        max_z = np.max(self.vertices[:, 2])

        self.x_range = np.array([min_x, max_x])
        self.y_range = np.array([min_y, max_y])
        self.z_range = np.array([min_z, max_z])

    def find_hotspots(self, n_hotspots):
        """
        Find hotspots based on heatmap values and clustering.
        """
        max_heat = np.max(self.heatmap)

        # Identify locations with high heat values
        where_is_heat = np.argwhere(self.heatmap >= 0.95 * max_heat)
        
        # Apply KMeans clustering to find hotspot centers
        kmeans = KMeans(n_clusters=n_hotspots, random_state=42).fit(self.grid_points[where_is_heat][:, 0])

        cluster_centers = kmeans.cluster_centers_

        self.highest_points = []

        for cluster in cluster_centers:
            # Find the closest grid point to the cluster center
            distances = np.linalg.norm(self.grid_points - cluster, axis=1)
            min_index = np.argmin(distances)
            self.highest_points.append(self.find_highest_point_above(self.grid_points[min_index], np.argwhere(self.up_vec == 1)[0][0]))
            
        self.highest_points = np.array(self.highest_points)
    
    def find_highest_point_above(self, given_point, dimension):
        """
        Find the highest point above a given point in the specified dimension.
        Currently supports only z direction.
        """
        gx, gy = given_point[0], given_point[1]
        
        # Filter points with the same x and y coordinates
        same_xy_points = self.grid_points[(self.grid_points[:, 0] == gx) & (self.grid_points[:, 1] == gy)]
        
        if same_xy_points.size == 0:
            return None

        # Return the point with the maximum z value
        highest_point = same_xy_points[np.argmax(same_xy_points[:, 2])]

        return highest_point

    def generate_cylindrical_feeders(self, radius, height, num_points_surface=10):
        """
        Generate cylindrical feeders based on hotspot points.
        """
        self.feeder_points = []

        for h in self.highest_points:
            # Unpack the hotspot coordinates
            x0, y0, z0 = h
            
            # Generate cylindrical surface points
            theta = np.linspace(0, 2 * np.pi, num_points_surface)
            z_surface = np.linspace(z0, z0 + height, num_points_surface)
            theta_grid, z_grid = np.meshgrid(theta, z_surface)
            
            x_surface = x0 + radius * np.cos(theta_grid)
            y_surface = y0 + radius * np.sin(theta_grid)
            
            surface_points = np.vstack((x_surface.ravel(), y_surface.ravel(), z_grid.ravel())).T

            for i in surface_points:
                self.feeder_points.append(i)

        self.feeder_points = np.array(self.feeder_points)

    def assign_heat(self, point_idx):
        """
        Assign heat value to a point based on a predefined formula.
        """
        point = self.grid_points[point_idx]
        self.heatmap[point_idx] = point[0]**2 + point[2]**2 - point[1]**2

    def create_heat_map(self, resolution):
        """
        Create a heatmap based on the grid points and heat assignment.
        """
        x = np.linspace(self.x_range[0], self.x_range[1], resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], resolution)
        z = np.linspace(self.z_range[0], self.z_range[1], resolution)
        
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        self.grid_points = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

        self.middle = np.array([x[int(len(x)/2):int(len(x)/2) + 1], y[int(len(y)/2):int(len(y)/2) + 1], z[int(len(z)/2):int(len(z)/2) + 1]])
        self.heatmap = np.zeros(len(self.grid_points[:, 0]))

        for i in range(len(self.heatmap)):
            self.assign_heat(i)

"""# Example usage. 
feedfinder = FindMyFeeders(10, file="models/newscene.obj", up_vec=np.array([0, 0, 1]))
feedfinder.create_heat_map(10)
feedfinder.find_hotspots(2)
feedfinder.generate_cylindrical_feeders(10, 100)
feedfinder.show_heatmap()
"""