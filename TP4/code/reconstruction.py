#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    # YOUR CODE
    x_x = np.linspace(min_grid[0], min_grid[0] + size_voxel*grid_resolution, grid_resolution)
    x_y = np.linspace(min_grid[1], min_grid[1] + size_voxel*grid_resolution, grid_resolution)
    x_z = np.linspace(min_grid[2], min_grid[2] + size_voxel*grid_resolution, grid_resolution)

    kdt = KDTree(points, leaf_size=30, metric='euclidean')

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                p = np.array([x_x[i], x_y[j], x_z[k]]) 

                dist, ind = kdt.query(p.reshape(1, -1), k=1)
                closest_point = points[ind.flatten()[0]]
                normal = normals[ind.flatten()[0]]

                vec_p = p - closest_point
                scalar_field[i, j, k] = np.dot(vec_p, normal)
    return scalar_field
    

# IMLS surface reconstruction
def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):
    # YOUR CODE
    x_x = np.linspace(min_grid[0], min_grid[0] + size_voxel*grid_resolution, grid_resolution)
    x_y = np.linspace(min_grid[1], min_grid[1] + size_voxel*grid_resolution, grid_resolution)
    x_z = np.linspace(min_grid[2], min_grid[2] + size_voxel*grid_resolution, grid_resolution)

    kdt = KDTree(points, leaf_size=30, metric='euclidean')

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                p = np.array([x_x[i], x_y[j], x_z[k]])

                dist, ind = kdt.query(p.reshape(1, -1), k=knn)
                dist = dist.flatten()
                ind = ind.flatten()

                vec_p = points[ind] - p

                w = np.exp(-dist**2 / 0.01)
                w /= np.sum(w) 

                scalar_field[i, j, k] = np.sum(w * np.einsum('ij,ij->i', vec_p, normals[ind]))

    return scalar_field



if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = 'TP4/data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128 #128
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
    print("size_voxel: ", size_voxel)
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

	# Compute the scalar field in the grid
    compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    #compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel,size_voxel,size_voxel))
    verts += min_grid
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj='bunny_mesh_hoppe_128.ply', file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	


