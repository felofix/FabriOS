# Importing various packages. 
import numpy as np # Package for array manipulation.
import matplotlib.pyplot as plt # Plotting package.
import pywavefront # To extract vertices and faces from the obj file.
from sklearn.cluster import KMeans # Finding close neighbours. 
from scipy.spatial.distance import cdist # Finding close neighbours.
from scipy.spatial import KDTree # Finding close neighbours.
from sklearn.neighbors import KDTree as KDTreetwo # Finding close neighbours.
import time # Taking time. 
from collections import Counter # Counter. 
import random 
import csv
import pandas as pd
from datetime import datetime

class Particle:
	def __init__(self, x, y, z, vx, vy, vz):
		"""
		Particle class that holds all relevant information for 
		each particle.
		"""
		self.x = x
		self.y = y 
		self.z = z 
		self.vx = vx
		self.vy = vy 
		self.vz = vz

	def update_position(self, x, y, z):
		"""
		Updates position of the particle. 
		"""
		self.x = x 
		self.y = y 
		self.z = z

def ray_intersect_triangle(p0, p1, v1, v2, v3):
	"""Möller–Trumbore ray-triangle intersection algorithm
	that finds out if a ray intersects with a triangle or not. 
	This is just a mathematical formula. 
	"""
	EPSILON = 1e-9
	edge1 = v2 - v1
	edge2 = v3 - v1
	h = np.cross(p1 - p0, edge2)
	a = np.dot(edge1, h)
	if -EPSILON < a < EPSILON:
		return None  # This ray is parallel to this triangle.
	f = 1.0 / a
	s = p0 - v1
	u = f * np.dot(s, h)
	if u < 0.0 or u > 1.0:
		return None
	q = np.cross(s, edge1)
	v = f * np.dot(p1 - p0, q)
	if v < 0.0 or u + v > 1.0:
		return None
	t = f * np.dot(edge2, q)
	if t > EPSILON:
		intersection_point = p0 + t * (p1 - p0)
		return intersection_point  # Return the intersection point
	else:
		return None  # This means that there is a line intersection but not a ray intersection.

class FindMyVents:
	"""
	The vent class that holds all relevent information 
	about the model where the vents are to be located. 
	"""
	def __init__(self, dt, steps, file=None, up_vec=np.array([0, 1, 0]), radius=2):
		"""
		Initializing all of relevent information about the file. 
		"""

		self.file = file # Filename. 
		self.up_vec = up_vec # Up orientation for vector. 
		self.dt = dt # Timestep. 
		self.steps = steps # How many steps?
		self.radius = radius # Radius which is part of various parts of the program. 

		if len(file[0]) > 1: # Figures out if a file is included or not. 
			self.vertices = file[0]
			self.faces = file[1]
			self.normals = []
		else:
			self.obj_to_point()

		self.find_dimensions()

		centroids = [] # Centers of eaach of the facese. 

		for face in self.faces:
			if len(face) == 3:  # Triangle
				v1, v2, v3 = self.vertices[face]
				centroids.append(self.centroid_of_triangle(v1, v2, v3))

		self.centroids = np.array(centroids)

		self.radius = np.max([self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0], self.x_range[1] - self.x_range[0]])/10

		self.original_vertices = self.vertices # Storing original vertices. 

	def find_dimensions(self):
		"""
		Finds a box around the model. 
		"""
		min_x = np.min(self.vertices[:, 0])
		max_x = np.max(self.vertices[:, 0])
		min_y = np.min(self.vertices[:, 1])
		max_y = np.max(self.vertices[:, 1])
		min_z = np.min(self.vertices[:, 2])
		max_z = np.max(self.vertices[:, 2])

		self.x_range = np.array([min_x, max_x])
		self.y_range = np.array([min_y, max_y])
		self.z_range = np.array([min_z, max_z])

	def obj_to_point(self):
		"""
		Creating vertices and faces from the obj file, so that 
		it can be used in this program. 
		"""
		file = self.file
		scene = pywavefront.Wavefront(file, create_materials=True, collect_faces=True)
		points = np.array(scene.vertices)
		self.vertices = points
		self.faces = scene.mesh_list[0].faces

	def ray_hits_face(self, p0, p1):
		"""
		Determins if a ray hits any face in the model. 
		"""
		faces = self.faces
		vertices = self.vertices

		for i, face in enumerate(faces):
			v1 = np.array(vertices[face[0]])
			v2 = np.array(vertices[face[1]])
			v3 = np.array(vertices[face[2]])

			intersection = ray_intersect_triangle(p0, p1, v1, v2, v3)

			if intersection is not None:
				return i, intersection  # Return the index of the hit face and the intersection point

		return None, None

	def centroid_of_triangle(self, v1, v2, v3):
		"""
		Finds the center of a face. 
		"""
		return (v1 + v2 + v3) / 3

	def hit_face(self, particle, faceidx, intersection_point):
		"""
		This essentialy figures out how the ray is updated
		if a ray hits a new face such that it follows this new direction. 
		"""		

		direction_vector = np.array([particle.vx, particle.vy, particle.vz])

		face = self.faces[faceidx]
		vertices = self.vertices
		v1 = np.array(vertices[face[0]])
		v2 = np.array(vertices[face[1]])
		v3 = np.array(vertices[face[2]])

		normal = np.cross(v2 - v1, v3 - v1)
		normal = normal / np.linalg.norm(normal)  # Ensure the normal is a unit vector

		# Reflect the velocity in the plane of the face
		reflected_vel = direction_vector - 2 * np.dot(direction_vector, normal) * normal

		# Project the reflected velocity onto the plane
		plane_normal = np.cross(v2 - v1, v3 - v1)
		plane_normal = plane_normal / np.linalg.norm(plane_normal)
		plane_component = reflected_vel - np.dot(reflected_vel, plane_normal) * plane_normal

		# Check if the plane component has the correct upward direction
		if np.dot(plane_component, self.up_vec) < 0:
			# Rotate by pi in the plane
			plane_component = -plane_component

		plane_component /= np.linalg.norm(plane_component)

		particle.vx = plane_component[0]*20 # This needs to be changd and just randomly set. 
		particle.vy = plane_component[1]*20
		particle.vz = plane_component[2]*20

		# Update particle position to the intersection point
		particle.x = intersection_point[0]
		particle.y = intersection_point[1]
		particle.z = intersection_point[2]

	def plot_shape_and_path(self, path):
		"""
		Plots path of one particle, should be a list of 
		x, y and z coordinates. 
		"""
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.scatter3D(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2] ,color="black", label='shape', alpha=0.3)
		ax.scatter3D(path[:, 0], path[:, 1], path[:, 2], color="red", label='particle')

		plt.show()

	def show_shape(self):
		"""
		Show the shape of the model. 
		"""
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.scatter3D(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2] ,color="black", label='shape', alpha=0.3)

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		plt.show()

	def top_point_face(self, faceidx):
		"""
		This located the highest point in a face with regards 
		to the up direction. 
		"""
		face = self.faces[faceidx]

		if self.up_vec[0] == 1:
			maxxer = np.argmax(self.vertices[face][:, 0])
			return self.vertices[face][maxxer]
		if self.up_vec[1] == 1:
			maxxer = np.argmax(self.vertices[face][:, 1])
			return self.vertices[face][maxxer]
		if self.up_vec[2] == 1:
			maxxer = np.argmax(self.vertices[face][:, 2])
			return self.vertices[face][maxxer]

	def create_particles(self, n):
		"""
		Creates the particle to be simulated and their
		x, y and z coordinates. 
		"""
		particles = np.empty((n, 3))
		particles[:, 0] = np.random.uniform(self.x_range[0], self.x_range[1], n)
		particles[:, 1] = np.random.uniform(self.y_range[0], self.y_range[1], n)
		particles[:, 2] = np.random.uniform(self.z_range[0], self.z_range[1], n)
		return particles

	def create_directions(self, n):
		# Generate random perturbations
		perturbations = np.random.normal(scale=0.1, size=(n, 3))

		# Generate random directions close to the main direction
		random_directions = self.up_vec + perturbations

		# Normalize the directions
		random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]

		return random_directions

	def faces_containing_vertex(self, vertex_index):
		# Check which faces contain the vertex
		return np.nonzero(self.faces == vertex_index)[0]

	def is_flat(self, top_point):
		"""
		Checks if a face is flat or not. 
		"""
		matches = np.all(self.vertices == top_point, axis=1)

		# Find the indices where the rows match
		indx = np.where(matches)[0]

		containing_faces = self.faces_containing_vertex(indx)

		for faceidx in containing_faces:
			face = self.faces[faceidx]
			vertices = self.vertices
			v1 = np.array(vertices[face[0]])
			v2 = np.array(vertices[face[1]])
			v3 = np.array(vertices[face[2]])

			normal = np.cross(v2 - v1, v3 - v1)
			normal = normal / np.linalg.norm(normal)  # Ensure the normal is a unit vector

			dot_product = np.dot(normal, self.up_vec)

			is_parallel = np.abs(np.abs(dot_product) - 1.0) <= 0.01

			if is_parallel:
				return True
			
		return False

	def close_point(self, particle):
		"""
		Finds the closest point to a particle. 
		"""
		tree = KDTree(self.centroids)

		indices = tree.query_ball_point(np.array([particle.x, particle.y, particle.z]), self.radius)

		points_within_radius = self.centroids[indices] 

		if len(points_within_radius) == 0:
			return False

		if self.up_vec[0] == 1:
			maxxer = np.argmax(points_within_radius[:, 0])
			particle.x = points_within_radius[maxxer][0]
			particle.y = points_within_radius[maxxer][1]
			particle.z = points_within_radius[maxxer][2]

			return True

		if self.up_vec[1] == 1:
			maxxer = np.argmax(points_within_radius[:, 1])
			particle.x = points_within_radius[maxxer][0]
			particle.y = points_within_radius[maxxer][1]
			particle.z = points_within_radius[maxxer][2]

			return True

		if self.up_vec[2] == 1:
			maxxer = np.argmax(points_within_radius[:, 2])

			if points_within_radius[maxxer][2] > particle.z:
				# Updates position. 
				particle.x = points_within_radius[maxxer][0]
				particle.y = points_within_radius[maxxer][1]
				particle.z = points_within_radius[maxxer][2]
				return True

			else:
				return False

	def face_smudge(self, particle):
		"""
		Smudging for easthtics. 
		"""
		index = np.where(np.all(self.centroids == np.array([particle.x, particle.y, particle.z]), axis=1))[0]
		if len(index) == 0:
			return None

		face = self.faces[index[0]]
		vertices = self.vertices
		v1 = np.array(vertices[face[0]])
		v2 = np.array(vertices[face[1]])
		v3 = np.array(vertices[face[2]])

		r1 = np.sqrt(np.random.rand())
		r2 = np.random.rand()
		point = (1 - r1) * v1 + r1 * (1 - r2) * v2 + r1 * r2 * v3
		
		return point
			

	def simulate(self, particle):
		"""
		Actually simulates the path of the particle. 
		"""
		positions = [] # Storage of all the positions of the particle. 
		dt = self.dt 
		steps = self.steps

		positions.append([particle.x, particle.y, particle.z]) # Appends the starting poistion. 
		lastface = None 

		for i in range(1, steps):
			"""
			For all steps. 
			"""
			p0 = np.array([particle.x, particle.y, particle.z])
			p1 = p0 + dt * np.array([particle.vx, particle.vy, particle.vz]) # Create an extended ray. 
			hit_face_idx, intersection_point = self.ray_hits_face(p0, p1) # Check if ray hits a face. 

			if hit_face_idx == None and lastface == None: # Line didnt hit anything from start.
				return None

			if hit_face_idx is not None: # Update lastface and position. 
				self.hit_face(particle, hit_face_idx, intersection_point)
				lastface = hit_face_idx

			else:
				while self.close_point(particle) == True:
					positions.append([particle.x, particle.y, particle.z])
					continue

				top = self.top_point_face(lastface)

				if self.is_flat(top): # Last face is flat. 
					return None

				smudge = self.face_smudge(particle)

				if smudge is not None:
					positions.append(smudge)

				return np.array(positions)


			particle.update_position(intersection_point[0], intersection_point[1], intersection_point[2])
			positions.append([particle.x, particle.y, particle.z])

	def rotate_vertices(self, dimension, theta, extra_points = []):
		"""
		Rotates the vertices according to a given theta. 
		"""
		theta_rad = np.radians(theta)  # Convert degrees to radians

		if dimension == 'x':
			R = np.array([[1, 0, 0],
						  [0, np.cos(theta_rad), -np.sin(theta_rad)],
						  [0, np.sin(theta_rad), np.cos(theta_rad)]])	

		if dimension == 'y':
			R = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
						  [0, 1, 0],
						  [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
			

		if dimension == 'z':
			R = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
						  [np.sin(theta_rad), np.cos(theta_rad), 0],
						  [0, 0, 1]])

		self.vertices = self.vertices @ R.T
		self.centroids = self.centroids @ R.T

		if len(extra_points) != 0:
			return extra_points @ R.T

	def find_vents(self, n, nvents):
		"""
		This does the simulation for each of the particles. 
		"""
		particles = self.create_particles(n)
		directions = self.create_directions(n)
		tops = []

		for p in range(len(particles)):
			particle = Particle(particles[p][0], particles[p][1], particles[p][2], directions[p][0], directions[p][1], directions[p][2])
			position = self.simulate(particle)

			if position is not None:
				tops.append(list(position[-1]))


		vertices_tuples = [tuple(vertex) for vertex in np.array(tops)]

		# Count occurrences of each vertex
		vertex_counts = Counter(vertices_tuples)

		# Find the vertex with the maximum count
		most_common_vertices = vertex_counts.most_common(nvents)

		toppers = []

		for vertex, count in most_common_vertices:
			toppers.append(vertex)

		self.tops = np.array(toppers)

	def show_vents(self):
		"""
		Shows vents. 
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.scatter3D(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2] ,color="black", label='shape', alpha=0.25)
		ax.scatter3D(self.tops[:, 0], self.tops[:, 1], self.tops[:, 2], color="red", label='Top')

		plt.show()

	def avg_number_neighbours(self):
		"""
		Computes the average number of neightbours within some vacinity. 
		"""
		if len(self.tops) == 0:
			return 100000
		tree = KDTreetwo(self.tops)
		neighbors = tree.query_radius(self.tops, r=self.radius)
		num_neighbors = [len(neighbor) - 1 for neighbor in neighbors]  # subtract 1 to exclude the point itself
		average_neighbors = np.mean(num_neighbors)/len(self.tops)

		return average_neighbors

	def optimize_vents(self, n_particles, n_vents, opacity = 4):
		"""
		n_particles = particles to try.
		n_vents = vents to find.
		This assumes up direction is z. 
		it does three iterations. 
		"""
		xranger = np.random.randint(0, 181, opacity)
		yranger = np.random.randint(0, 181, opacity)

		bestx = 0
		besty = 0
		best_fitness = 100000

		for i in range(opacity):
			for j in range(opacity): 
				self.rotate_vertices("x", xranger[i])
				self.rotate_vertices("y", yranger[j])

				self.find_vents(n_particles, n_vents)
				
				fitness = self.avg_number_neighbours()

				print(fitness)
				
				if fitness < best_fitness:
					bestx = xranger[i]
					besty = yranger[j]

		self.rotate_vertices("x", bestx)
		self.rotate_vertices("y", besty)
		self.find_vents(n_particles, n_vents)
		print(bestx, besty)

	def create_all_files(self):
		"""
		Creating files.
		"""
		vertices, other_lines = read_obj(self.file)

		now = datetime.now()
		timestamp = now.strftime("%Y%m%d_%H%M%S")
		filename_one = f"data/points_{timestamp}.csv"
		filename_two = f"data/obj_{timestamp}.obj"

		write_obj(filename_two, self.vertices, other_lines)

		magnitudes = self.magnitudes()

		self.tops = np.hstack((self.tops, magnitudes))

		self.create_files(filename_one)

	def magnitudes(self):
		"""
		Magnitudes. 
		"""

		tree = KDTree(self.tops)
		magnitudes = np.zeros((len(self.tops), 1))

		for i in range(len(self.tops)):
			indices = tree.query_ball_point(np.array([self.tops[i][0], self.tops[i][1], self.tops[i][2]]), self.radius)
			magnitudes[i] = len(indices)

		print(magnitudes)

		return magnitudes
			

	def create_files(self, filename):
		"""
		Makes files. 
		"""
		# Write to CSV file
		df = pd.DataFrame(self.tops, columns=['X', 'Y', 'Z', 'Magnitude'])
		df.to_csv(filename, index=False)

		print(f"CSV file '{filename}' created successfully.")


def read_obj(file_path):
	"""
	Reads and writes and object file. 
	"""
    vertices = []
    other_lines = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            else:
                other_lines.append(line)
    
    return vertices, other_lines

def write_obj(file_path, vertices, other_lines):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for line in other_lines:
            file.write(line)



"""# Example usage:
ventfinder = FindMyVents(10, 1000, file="models/Crank.obj", up_vec=np.array([0, 0, 1]))
ventfinder.find_vents(5000, 100)
ventfinder.create_all_files()
"""



