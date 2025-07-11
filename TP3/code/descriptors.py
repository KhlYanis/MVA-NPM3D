#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points):

    # Compute the barycenter, then the centered points
    barycenter = np.mean(points, axis = 0)
    centered_points = points - barycenter

    # Compute the covariance matrix 
    cov_matrix = (1/points.shape[0]) * np.dot(centered_points.T, centered_points)

    # Compute the eigenvalues/eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius = None, k = None):
    assert (radius is not None) or (k is not None), "Either radius or k has to be set to an integer value"
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((cloud.shape[0], 3))
    all_eigenvectors = np.zeros((cloud.shape[0], 3, 3))

    tree = KDTree(query_points) # Default value of leaf_size is 40

    for i, query_point in enumerate(query_points):
        if k is None : 
            idx_neighbors = tree.query_radius(query_point.reshape(1, -1) , r = radius)[0]
        else : 
            idx_neighbors = tree.query(query_point.reshape(1, -1), k, return_distance = False)[0]

        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[idx_neighbors])

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    # Compute the eigenvalues and eigenvects of each query point neighborhood
    eigenvals, eigenvects = compute_local_PCA(query_points, cloud_points, radius)

    n = eigenvects[:, :, 0]
    eps = 1e-8

    # Compute the different quantities
    verticality = 2*np.arcsin(np.abs(n[..., 2]) / np.pi)
    linearity = 1 - (eigenvals[:, 1]/(eigenvals[:, 2] + eps))
    planarity = (eigenvals[:, 1] - eigenvals[:, 0]) / (eigenvals[:, 2] + eps)
    sphericity = eigenvals[:, 0] / (eigenvals[:, 2] + eps)

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = 'TP3\data\Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    # with radius = 0.5m
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = 'TP3/data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud with k=30
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, radius = 0.5)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('TP3/Lille_street_small_normals_radius.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    # with k = 30
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = 'TP3/data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud with k=30
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, k = 30)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('TP3/Lille_street_small_normals_k30.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    ### Bonus section :
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = 'TP3/data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute the features for a radius of 0.5m
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, radius = 0.5)

        # Save cloud with scalars
        write_ply('TP3/Lille_street_small_bonus.ply', (cloud, verticality, linearity, planarity, sphericity),
                                                ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
        