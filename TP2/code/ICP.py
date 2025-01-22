#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0],1))

    # Calcul des barycentres
    data_barycenter = np.mean(data, axis = 1, keepdims = True)
    ref_barycenter = np.mean(ref, axis = 1, keepdims = True)

    # Centrage des nuages de points par rapport à leur barycentre
    data_centered = data - data_barycenter
    ref_centered = ref - ref_barycenter

    # Calcul de la matrice de covariance
    H = data_centered.dot(ref_centered.T)

    # Calcul de la décomposition en valeurs singulières (SVD) de H
    U, _, Vt = np.linalg.svd(H)

    # Calcul de la matrice de rotation
    R = (Vt.T).dot(U.T)
    # Calcul du vecteur de translation
    T = ref_barycenter - R.dot(data_barycenter)
    
    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE


    kd_tree = KDTree(ref.T)
    for i in range(max_iter):
        _, neighbors = kd_tree.query(data_aligned.T, k=1)
        neighbors = neighbors.squeeze()
        R, T = best_rigid_transform(data_aligned, ref[:,neighbors])
        data_aligned = R @ data_aligned + T
        
        if i == 0:
            T_list.append(T)
            R_list.append(R)
        else:
            T_list.append(R @ T_list[-1] + T)
            R_list.append(R @ R_list[-1])

        neighbors_list.append(neighbors)

        
        rms = np.sqrt(np.mean(np.sum((data_aligned - ref[:,neighbors].squeeze())**2, axis=0)))
        RMS_list.append(rms)
    
        if rms < RMS_threshold:
            break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        sampling_limit = number of random points to select at each iteration
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    kd_tree = KDTree(ref.T)
    for i in range(max_iter):
        data_random = data_aligned[:,np.random.choice(data_aligned.shape[1], sampling_limit, replace=False)]
        _, neighbors = kd_tree.query(data_random.T, k=1)
        neighbors = neighbors.squeeze()
        R, T = best_rigid_transform(data_random, ref[:,neighbors])
        data_aligned = R @ data_aligned + T

        if i%5 == 0:
            print('Iteration', i)
        
        if i == 0:
            T_list.append(T)
            R_list.append(R)
        else:
            T_list.append(R @ T_list[-1] + T)
            R_list.append(R @ R_list[-1])

        neighbors_list.append(neighbors)

        _, all_neighbors = kd_tree.query(data_aligned.T, k=1)
        #rms = np.sqrt(np.mean(np.sum((data_random - ref[:,neighbors].squeeze())**2, axis=0)))
        rms = np.sqrt(np.mean(np.sum((data_aligned - ref[:,all_neighbors].squeeze())**2, axis=0)))
        RMS_list.append(rms)
    
        if rms < RMS_threshold:
            break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = 'TP2/data/bunny_original.ply'
        bunny_r_path = 'TP2/data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('TP2/data/bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = 'TP2/data/ref2D.ply'
        data2D_path = 'TP2/data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = 'TP2/data/bunny_original.ply'
        bunny_p_path = 'TP2/data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()

    # Bonus
    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        NDC_o_path = 'TP2/data/Notre_Dame_Des_Champs_1.ply'
        NDC_p_path = 'TP2/data/Notre_Dame_Des_Champs_2.ply'
        
        # Load clouds
        NDC_o_ply = read_ply(NDC_o_path)
        NDC_p_ply = read_ply(NDC_p_path)
        NDC_o = np.vstack((NDC_o_ply['x'], NDC_o_ply['y'], NDC_o_ply['z']))
        NDC_p = np.vstack((NDC_p_ply['x'], NDC_p_ply['y'], NDC_p_ply['z']))

        # Apply ICP
        #NDC_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(NDC_p, NDC_o, 20, 1e-4, sampling_limit=1000)
        NDC_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(NDC_p, NDC_o, 20, 1e-4, sampling_limit=15000)
        
        # Show ICP
        # show_ICP(NDC_p, NDC_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()