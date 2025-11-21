# Final project
# Miguel Gutierrez
# cube_in_chessboard_video.py
# Projective Geometry 2025

import gepr_math as gpmath

import cv2
import numpy as np

def main():
    # Load all frames in a list
    raw_frames = []
    for i in range(1, 735):
        path = f"resources/images/img_{i:03d}.jpg"
        raw_frames.append(cv2.imread(path))

    # Calculate frame size
    frame_size = (raw_frames[0].shape[1], raw_frames[0].shape[0])

    # Chess board data
    rows, cols = 6, 9
    pattern_size = (cols, rows)
    square_size = 4

    # Load camera data
    K = np.loadtxt('resources/K.txt')

    # Cube data
    cube_vertices = np.array([
    [0, 0, 0],
    [8, 0, 0],
    [8, 8, 0],
    [0, 8, 0],
    [0, 0, 8],
    [8, 0, 8],
    [8, 8, 8],
    [0, 8, 8]])

    # Fill a (rows, cols, 2) np array multiplying the position by square size
    i_indices, j_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    i_indices = i_indices[..., np.newaxis]
    j_indices = j_indices[..., np.newaxis]
    real_chess_corners = np.concatenate((square_size * i_indices, square_size * j_indices), axis=2)
    real_chess_corners_reshaped = real_chess_corners.reshape(-1, 2)

    # Create video writer
    out_mp4 = cv2.VideoWriter("resultados/parte4.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, frame_size)

    for i in range(0, len(raw_frames), 1):
        raw_frame = raw_frames[i]
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
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
            
            # Axes in image
            cube_img = raw_frame.copy()
            
            R, t = gpmath.get_camera_pose(homography_mat, K)
            R = gpmath.correct_rotation_matrix(R)

            # Make t column vector
            t = t.reshape((3,1))

            RT = np.hstack((R, t))
            P = np.dot(K, RT)

            draw_cube(cube_img, cube_vertices, P)

            out_mp4.write(cube_img)
        else:
            print(f"Chessboard not found in frame {i}\nAdding raw frame to video")
            out_mp4.write(raw_frame)

        percentaje = i / (len(raw_frames) - 1) * 100
        print(f"Frame {i} completed - {int(percentaje)}%")

    out_mp4.release()
    print("Video completed!")

def draw_cube(img, cube_data, projection_matrix):
    v0 = transform_to_image_plane(cube_data[0], projection_matrix)
    v1 = transform_to_image_plane(cube_data[1], projection_matrix)
    v2 = transform_to_image_plane(cube_data[2], projection_matrix)
    v3 = transform_to_image_plane(cube_data[3], projection_matrix)
    v4 = transform_to_image_plane(cube_data[4], projection_matrix)
    v5 = transform_to_image_plane(cube_data[5], projection_matrix)
    v6 = transform_to_image_plane(cube_data[6], projection_matrix)
    v7 = transform_to_image_plane(cube_data[7], projection_matrix)

    # Draw lower square
    main_color = (28, 114, 235)
    cv2.line(img, v0, v1, main_color, thickness=2)
    cv2.line(img, v0, v3, main_color, thickness=2)
    cv2.line(img, v2, v1, main_color, thickness=2)
    cv2.line(img, v2, v3, main_color, thickness=2)

    # Draw upper square
    cv2.line(img, v4, v5, main_color, thickness=2)
    cv2.line(img, v4, v7, main_color, thickness=2)
    cv2.line(img, v6, v5, main_color, thickness=2)
    cv2.line(img, v6, v7, main_color, thickness=2)

    # Join squares
    cv2.line(img, v0, v4, main_color, thickness=2)
    cv2.line(img, v1, v5, main_color, thickness=2)
    cv2.line(img, v2, v6, main_color, thickness=2)
    cv2.line(img, v3, v7, main_color, thickness=2)

def transform_to_image_plane(point3d, projection_matrix):
    point4d = np.append(point3d, 1)
    image_coordinates = np.dot(projection_matrix, point4d)
    return (image_coordinates[:2] / image_coordinates[2]).astype(int)

if __name__ == "__main__":
    main()