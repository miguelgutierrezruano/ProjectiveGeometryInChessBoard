# Final project
# Miguel Gutierrez
# gepr_math.py
# Projective Geometry 2025

import numpy as np

# Direct Linear Transformation
def compute_homography(src_points, dst_points):

    matrices = []

    for src, dst in zip(src_points, dst_points):
        M = get_dlt_2_9_matrix(src, dst)
        matrices.append(M)
    
    A = np.vstack(matrices)
    
    U, S, Vt = np.linalg.svd(A)
    # Get last row
    h = Vt[-1, :] 
    H = h.reshape(3, 3)

    # Normalize so H[2,2] = 1
    H = H / H[2, 2]

    return H

def get_dlt_2_9_matrix(point1, point2):

    return np.array([
        [point1[0], point1[1], 1,         0,         0, 0, -point1[0] * point2[0], -point1[1] * point2[0], -point2[0]],
        [        0,         0, 0, point1[0], point1[1], 1, -point1[0] * point2[1], -point1[1] * point2[1], -point2[1]]
    ])

def get_camera_pose(H, K):
    # Get the K inverted
    kInverted = np.linalg.inv(K)

    # Multiply K inverted and the homography (r1r2t columns)
    M = np.dot(kInverted, H)

    # Compute lambda value: Inverse of the norm of the first rotation vector
    lambda_value = 1 / np.linalg.norm(M[:, 0])

    # Normalize M with negative lambda value
    M = -lambda_value * M

    # Extract individual columns
    r1 = M[:, 0]
    r2 = M[:, 1]
    t  = M[:, 2]

    # Compute r3 as the scalar of r1 and r2
    r3 = np.cross(r1, r2)

    # Create rotation matrix
    R = np.column_stack((r1, r2, r3))

    return R, t

def is_rotation_matrix(R):
    # Check if the matrix determinant is 1
    det = np.linalg.det(R)

    return np.isclose(det, 1.0)

def correct_rotation_matrix(R):
    # Fix rotation matrix with singular value decomposition
    U, _, Vt = np.linalg.svd(R)
    R_corrected = np.dot(U, Vt)

    return R_corrected