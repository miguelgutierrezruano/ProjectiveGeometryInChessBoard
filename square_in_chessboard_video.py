# Final project
# Miguel Gutierrez
# square_in_chessboard_video.py
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

    # Fill a (rows, cols, 2) np array multiplying the position by square size
    i_indices, j_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    i_indices = i_indices[..., np.newaxis]
    j_indices = j_indices[..., np.newaxis]
    real_chess_corners = np.concatenate((square_size * i_indices, square_size * j_indices), axis=2)
    real_chess_corners_reshaped = real_chess_corners.reshape(-1, 2)

    # Create video writer
    out_mp4 = cv2.VideoWriter("resultados/parte2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, frame_size)

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
            axes_img = raw_frame.copy()
            
            # Transform origin
            origin  = apply_homography_to_point(homography_mat, point=(real_chess_corners[0][0]))
            xValue  = apply_homography_to_point(homography_mat, point=(real_chess_corners[0][2]))
            yValue  = apply_homography_to_point(homography_mat, point=(real_chess_corners[2][0]))
            xyValue = apply_homography_to_point(homography_mat, point=(real_chess_corners[2][2]))

            cv2.line(axes_img, origin, xValue,  color=(0, 255, 0), thickness=2)
            cv2.line(axes_img, origin, yValue,  color=(0, 255, 0), thickness=2)
            cv2.line(axes_img, xValue, xyValue, color=(0, 255, 0), thickness=2)
            cv2.line(axes_img, yValue, xyValue, color=(0, 255, 0), thickness=2)

            out_mp4.write(axes_img)
        else:
            print(f"Chessboard not found in frame {i}\nAdding raw frame to video")
            out_mp4.write(raw_frame)

        percentaje = i / (len(raw_frames) - 1) * 100
        print(f"Frame {i} completed - {int(percentaje)}%")

    out_mp4.release()
    print("Video completed!")

def apply_homography_to_point(homography, point):
    point3d = np.array([point[0], point[1], 1])
    img_point = homography.dot(point3d)
    # Divide by w
    img_point = img_point[:2] / img_point[2]
    return img_point.astype(int)

if __name__ == "__main__":
    main()