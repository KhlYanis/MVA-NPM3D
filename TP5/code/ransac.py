#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np
import numpy.ma as ma

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    
    point_plane = np.zeros((3,1))
    normal_plane = np.zeros((3,1))
    
    # TODO:
    v_1 = points[1] - points[0]
    v_2 = points[2] - points[0]

    normal_plane = np.cross(v_1, v_2)
    normal_plane /= np.linalg.norm(normal_plane)

    point_plane = points[0]
    
    return point_plane, normal_plane



def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):
    
    indexes = np.zeros(len(points), dtype=bool)
    
    # TODO:
    distances = np.dot(points - pt_plane, normal_plane)/np.linalg.norm(normal_plane)
    indexes = np.abs(distances) < threshold_in

    return indexes



def RANSAC(points, nb_draws=100, threshold_in=0.1):
    
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    # TODO:
    nb_points = len(points)

    for _ in range(nb_draws):
        # Draw randomly three points
        draw = np.random.choice(nb_points, 3, replace = False)

        drawn_points = points[draw]
        point_plane, normal_plane = compute_plane(drawn_points)

        # Retrieve the 
        idx_in_plane = in_plane(points, 
                                pt_plane = point_plane, 
                                normal_plane = normal_plane,
                                threshold_in = threshold_in)
        
        votes = np.sum(idx_in_plane)

        # Change the the best plane fitting
        if votes > best_vote :
            best_pt_plane = point_plane
            best_normal_plane = normal_plane
            best_vote = votes
   
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):
    
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)
	
    # TODO:
    for id_plane in range(nb_planes):

        if len(remaining_inds) < 3 :
            break

        # Run the RANSAC algorithm
        best_pt_plane, best_normal_plane, _ = RANSAC(points[remaining_inds],
                                                    nb_draws = nb_draws,
                                                    threshold_in = threshold_in)
        
        # Retrieve the points in the plane
        idx_in_plane = in_plane(points[remaining_inds], 
                                pt_plane = best_pt_plane, 
                                normal_plane = best_normal_plane,
                                threshold_in = threshold_in)
        
        # Add the points in the current plane
        plane_inds = np.append(plane_inds, remaining_inds[idx_in_plane])
        plane_labels = np.append(plane_labels, np.repeat(id_plane, idx_in_plane.sum()))

        # Update the remaining indices
        remaining_inds  = remaining_inds[~idx_in_plane]

    return plane_inds, remaining_inds, plane_labels



#------------------------------------------------------------------------------------------
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
    file_path = 'TP5/data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('TP5/data/plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('TP5/data/remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('TP5/data/best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('TP5/data/remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.10
    nb_planes = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

    # Save the best planes and remaining points
    write_ply('TP5/data/best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('TP5/data/remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    ## ----- QUESTION 3 ----- ##
    print('\n--- 5) ---\n')

    # Path of the file
    file_path_aux = 'TP5/data/Lille_street_small.ply'

    # Load point cloud
    data_aux = read_ply(file_path_aux)

    # Concatenate data
    points = np.vstack((data_aux['x'], data_aux['y'], data_aux['z'])).T
    nb_points = len(points)

    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

    # Save the best planes and remaining points
    write_ply('TP5/data/best_planes_Lille.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('TP5/data/remaining_points_best_planes_Lille.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    print('Done')
    