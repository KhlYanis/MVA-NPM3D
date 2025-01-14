#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[0:-1:factor]
    decimated_colors = colors[0:-1:factor]
    decimated_labels = labels[0:-1:factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, colors, labels, voxel_size = 0.15):

    # Récupération des extrémités du nuage de points
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    # Arrondi par défaut pour les coordonnées minimales
    min_x, min_y, min_z = np.floor(min_x * 100)/100, np.floor(min_y * 100)/100, np.floor(min_z * 100)/100
    # Arrondi par excès pour les coordonnées maximales
    max_x, max_y, max_z = np.round(max_x, 2), np.round(max_y, 2), np.round(max_z, 2)

    # On définit une grille avec un pas de voxel_size
    X_axis = np.arange(min_x, max_x, voxel_size)
    Y_axis = np.arange(min_y, max_y, voxel_size)
    Z_axis = np.arange(min_z, max_z, voxel_size)

    # Initialisation des grilles de points
    point_grid = np.zeros([len(X_axis), len(Y_axis), len(Z_axis), 3], dtype = np.float32)
    color_grid = np.zeros_like(point_grid, dtype = np.uint8)

    # Initialisation des compteurs de points par voxel
    point_count = np.zeros([len(X_axis), len(Y_axis), len(Z_axis)])
    # Initialisation de la grille de labels et de couleurs
    label_grid = np.zeros_like(point_count)

    for point, color, label in zip(points, colors, labels) :
        idx_x = int((point[0] - min_x) / voxel_size)
        idx_y = int((point[1] - min_y) / voxel_size)
        idx_z = int((point[2] - min_z) / voxel_size)

        point_grid[idx_x, idx_y, idx_z] += point
        color_grid[idx_x, idx_y, idx_z] = color
        label_grid[idx_x, idx_y, idx_z] = label   

        point_count[idx_x, idx_y, idx_z] += 1
    
    # Moynnage des points et des couleurs
    with np.errstate(divide='ignore', invalid='ignore'):
        point_grid = np.divide(point_grid, point_count[..., None])
        # Remplacement des valeurs nan par 0
        point_grid[point_count == 0] = 0  

    return point_grid.reshape(-1, 3), color_grid.reshape(-1, 3), label_grid.reshape(-1)


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
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('./decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    
    print('Done')

    # Grid subsampling
    print('grid subsampling...')

    t0 = time.time()
    subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling(points, colors, labels, voxel_size = 0.15)
    t1 = time.time()

    print('grid subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('./subsampled.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])