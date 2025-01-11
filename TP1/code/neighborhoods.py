#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    # queries est de taille [N, 3] où N correspond au nombre de queries
    # supports est de taille [M, 3] où M est le nombre de points dans le nuage de points

    neighborhoods = None

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = None

    return neighborhoods





# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = './TP1/data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:
        np.random.seed(42)

        # Define the search parameters
        num_queries = 5000
        radius = 0.20

        # YOUR CODE
        max_leaf = 500
        step = 25
        leaf_size = np.arange(1, max_leaf, step)

        build_time, inference_time = [], []
        random_idx = np.random.choice(points.shape[0], num_queries, replace = False)

        for size in leaf_size :            
            # Creation of the KDTree
            t1 = time.time()
            tree = KDTree(data = points, leaf_size = size)
            t2 = time.time()

            build_time.append(t2 - t1)

            # Inference
            t1 = time.time()
            tree.query_radius(points[random_idx], r = radius)
            t2 = time.time() 

            inference_time.append(t2 - t1)

        plt.figure()
        plt.plot(leaf_size, build_time)
        plt.xlabel('Leaf size')
        plt.ylabel('Build Time [s]')
        plt.title(f' {num_queries} queries with r = {radius}')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(leaf_size, inference_time, label = "Inference time")
        plt.xlabel('Leaf size')
        plt.ylabel('Inference Time [s]')
        plt.title(f'{num_queries} queries with r = {radius}')
        plt.grid()
        plt.show()

        print(f"La valeur optimal de leaf_size pur l'inférence est {leaf_size[np.argmin(inference_time)]}")


        
    if True :
        # Retrieve the optimal leaf number
        optimal_leaf = 76
        
        r_values = np.arange(0.05, 1.0, 0.05)

        time_for_radius = []
        random_idx = np.random.choice(points.shape[0], size = 1000, replace = False)
        
        for radius in r_values :
            tree = KDTree(data = points, leaf_size = optimal_leaf)
            
            t0 = time.time()
            tree.query_radius(points[random_idx], r = radius)
            t1 = time.time()

            time_for_radius.append(t1-t0)

        plt.figure()
        plt.plot(r_values, time_for_radius)
        plt.xlabel("Radius [m]")
        plt.ylabel("Inference time [s]")
        plt.title(f"Inference time for {1000} queries in function of radius")
        plt.grid(True)
        plt.show()

        time_r_20 = time_for_radius[np.argwhere(r_values == 0.20)[0, 0]]
        total_time_r_20 = (points.shape[0]/num_queries)*time_r_20
        print(f"Le temps pour r = 20 cm est de {total_time_r_20} secondes")