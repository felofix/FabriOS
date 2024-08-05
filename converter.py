import numpy as np
import meshio
from fenics import * 
import pyvista as pv

def read_vtk_custom(file_path):
    vertices = []
    tetrahedrons = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Read the vertices
    vertices_start_index = lines.index('POINTS 1862 double\n') + 1
    vertices_end_index = lines.index('CELLS 9285 43279\n')
    for line in lines[vertices_start_index:vertices_end_index - 1]:
        x, y, z = map(float, line.split())
        vertices.append((x, y, z))
    
    # Read the tetrahedrons
    cells_start_index = vertices_end_index + 1
    for line in lines[cells_start_index:]:
        parts = list(map(int, line.split()))

        if len(parts) == 5:
            # Add tetrahedron, where parts[1:] are the vertex indices
            tetrahedrons.append(parts[1:])
        
        elif len(parts) == 0:
            break
    
    return vertices, tetrahedrons

# Example usage
file_path = 'meshes/sherrif.vtk'
vertices, tetrahedrons = read_vtk_custom(file_path)



# Create mesh using meshio
mesh = meshio.Mesh(
    points=vertices,
    cells={"tetra": tetrahedrons}
)

# Write the mesh to a file
meshio.write("meshes/mesh.xml", mesh)
