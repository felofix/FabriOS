from fenics import *
import time
import numpy as np
import pywavefront

file = "models/holebox.obj"
scene = pywavefront.Wavefront(file, create_materials=True, collect_faces=True)
points = np.array(scene.vertices)
vertices = points
domain_vertices = []

for i in vertices:
	domain_vertices.append(Point(i))

mesh = Mesh()
PolyhedralMeshGenerator.generate(mesh, domain_vertices, 0.25)

"""
T = 5.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = nz = 30
mesh = BoxMesh(Point(-2, -2, -2), Point(2, 2, 2), nx, ny, nz)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Define initial value
u_0 = Constant(10)
u_n = interpolate(u_0, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

# Variational form
F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile = File('heat_gaussian/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u, bc)

    # Save to file and plot solution
    vtkfile << (u, t)

    # Update previous solution
    u_n.assign(u)
 """