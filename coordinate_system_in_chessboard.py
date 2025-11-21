# Final project
# Miguel Gutierrez
# coordinate_system_in_chessboard.py
# Projective Geometry 2025

import gepr_math as gpmath

import cv2
import numpy as np

def calculate_camera_pose():
    # Read image
    img = cv2.imread("resources/images/img_001.jpg")

    # Read previously calculated homography
    H = np.loadtxt('resultados/parte1_H.txt')

    # Read input camera calibration matrix
    K = np.loadtxt('resources/K.txt')

    print("H matrix:\n", H, "\n")
    print("K matrix:\n", K, "\n") 

    R, t = gpmath.get_camera_pose(H, K)

    print(f"Is rotation matrix: {gpmath.is_rotation_matrix(R)}")
    print("Rotation matrix R:\n", R, "\n")
    print("Fixing rotation matrix...\n")
    R = gpmath.correct_rotation_matrix(R)
    print(f"Is rotation matrix: {gpmath.is_rotation_matrix(R)}")
    print("Rotation matrix R:\n", R, "\n")

    # Save rotation matrix
    np.savetxt("resultados/parte3_R.txt", R)

    print("Traslation vector t:\n", t, "\n")

    # Save translation vector
    np.savetxt("resultados/parte3_t.txt", t)

    # Convert rotation matrix to vector for cv2.drawFrameAxes function
    rvec, _ = cv2.Rodrigues(R)

    axisLenght = 8.0
    image_with_axes = cv2.drawFrameAxes(img, K, np.zeros((1, 5)), rvec, t, axisLenght)

    cv2.imshow("Frame Axes", image_with_axes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("resultados/parte3.png", image_with_axes)

calculate_camera_pose()