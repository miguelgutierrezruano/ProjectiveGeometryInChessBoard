# Final project
# Miguel Gutierrez
# squares_in_chessboard.py
# Projective Geometry 2025

import cv2
import numpy as np

import gepr_math as gpmath

def main():
    # Load image data
    PATH = "resources/images/img_001.jpg"
    board_img = cv2.imread(PATH)

    # Chess board data
    rows, cols = 6, 9
    pattern_size = (cols, rows)
    square_size = 4

    # Fill a (rows, cols, 2) np array multiplying the position by square size
    i_indices, j_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    i_indices = i_indices[..., np.newaxis]
    j_indices = j_indices[..., np.newaxis]
    real_chess_corners = np.concatenate((square_size * i_indices, square_size * j_indices), axis=2)
    real_chess_corners_reshaped = real_chess_corners.reshape(-1, 2)

    # Same as:
    # grid = []
    # for i in range(0, rows, 1):
    #     for j in range(0, cols, 1):
    #         grid.append([i * square_size, j * square_size])

    # grid = np.array(grid)
    # grid.reshape(-1, 2)
    # print(np.array_equal(grid, real_chess_corners_reshaped))

    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if found:
        # Refine corners
        corners_refined = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
        # Truncate corners refined to get pixel data
        corner_pixels = corners_refined.astype(int)

        # Divide corner pixels in groups of column number (9)
        corner_column_sets = corner_pixels.reshape(rows, cols, 2)

        # Invert order to get the origin bottom left
        image_plane_corners = corner_column_sets[::-1]

        image_plane_corners_reshaped = image_plane_corners.reshape(-1, 2)

        homography_mat = gpmath.compute_homography(real_chess_corners_reshaped, image_plane_corners_reshaped)
        print(homography_mat, "\n")

        # Save homography in file
        np.savetxt("resultados/parte1_H.txt", homography_mat)

        # Axes in image
        square_aligned_img = board_img.copy()
            
        # Transform origin
        origin  = apply_homography_to_point(homography_mat, point=(real_chess_corners[0][0]))
        xValue  = apply_homography_to_point(homography_mat, point=(real_chess_corners[0][2]))
        yValue  = apply_homography_to_point(homography_mat, point=(real_chess_corners[2][0]))
        xyValue = apply_homography_to_point(homography_mat, point=(real_chess_corners[2][2]))

        cv2.line(square_aligned_img, origin, xValue,  color=(0, 255, 0), thickness=2)
        cv2.line(square_aligned_img, origin, yValue,  color=(0, 255, 0), thickness=2)
        cv2.line(square_aligned_img, xValue, xyValue, color=(0, 255, 0), thickness=2)
        cv2.line(square_aligned_img, yValue, xyValue, color=(0, 255, 0), thickness=2)

        show_image_and_wait_input("Square aligned", square_aligned_img)
        cv2.imwrite("resultados/parte1_2.png", square_aligned_img)

        # Square half
        square_half_img = board_img.copy()

        half_square = square_size / 2
        origin_and_half  = real_chess_corners[0][0] + half_square
        xValue_and_half  = real_chess_corners[0][2] + half_square
        yValue_and_half  = real_chess_corners[2][0] + half_square
        xyValue_and_half = real_chess_corners[2][2] + half_square

        origin  = apply_homography_to_point(homography_mat, origin_and_half)
        xValue  = apply_homography_to_point(homography_mat, xValue_and_half)
        yValue  = apply_homography_to_point(homography_mat, yValue_and_half)
        xyValue = apply_homography_to_point(homography_mat, xyValue_and_half)

        cv2.line(square_half_img, origin, xValue,  color=(0, 255, 0), thickness=2)
        cv2.line(square_half_img, origin, yValue,  color=(0, 255, 0), thickness=2)
        cv2.line(square_half_img, xValue, xyValue, color=(0, 255, 0), thickness=2)
        cv2.line(square_half_img, yValue, xyValue, color=(0, 255, 0), thickness=2)

        show_image_and_wait_input("Square in half", square_half_img)
        cv2.imwrite("resultados/parte1_3.png", square_half_img)
    else:
        print(f"Could not find chessboard within input image: {PATH}")

def apply_homography_to_point(homography, point):
    # Add w dimension
    point3d = np.array([point[0], point[1], 1])
    # Multiply by homography
    img_point = homography.dot(point3d)
    # Divide by w to get pixel point
    img_point = img_point[:2] / img_point[2]
    return img_point.astype(int)

def show_image_and_wait_input(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()