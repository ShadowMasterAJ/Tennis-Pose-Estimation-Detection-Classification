import math
import cv2
import numpy as np

def get_player_direction(prev_positions, current_position, frames):
    if len(prev_positions) < frames:
        return "Standing"
    
    # Calculate moving averages
    window_size = min(frames, len(prev_positions))
    x_avg = sum(pos[0] for pos in prev_positions[-window_size:]) / window_size
    y_avg = sum(pos[1] for pos in prev_positions[-window_size:]) / window_size
    
    # Calculate displacement
    dx = current_position[0] - x_avg
    dy = current_position[1] - y_avg
    
    # Define thresholds for movement detection
    movement_threshold = 5.0  # Adjust this value based on your needs
    
    # Calculate the magnitude of movement
    magnitude = (dx**2 + dy**2)**0.5
    
    if magnitude < movement_threshold:
        return "Standing"
    
    # Calculate the angle of movement
    angle = math.atan2(dy, dx)
    
    # Define direction sectors (in radians)
    sectors = {
        "Right": (-math.pi/4, math.pi/4),
        "Backward": (math.pi/4, 3*math.pi/4),
        "Left": (3*math.pi/4, math.pi) or (-math.pi, -3*math.pi/4),
        "Forward": (-3*math.pi/4, -math.pi/4)
    }
    
    # Determine the direction based on the angle
    for direction, (start, end) in sectors.items():
        if start <= angle < end:
            return direction
    
    return "Standing"  # Default case

def draw_keypoints_and_skeleton(img, keypoints):
    joint_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
        (64, 64, 0), (64, 0, 64), (0, 64, 64)
    ]
    skeleton = [
        [1, 2], [1, 3], [1, 18],
        [2, 4],
        [3, 5],
        [6, 8], [6, 12], [6, 18],
        [7, 9], [7, 13], [7, 18],
        [8, 10],
        [9, 11],
        [12, 13], [12, 14],
        [13, 15],
        [14, 16],
        [15, 17]
    ]

    # Create a black frame
    black_frame = np.zeros_like(img)
    
    # Draw skeleton on black frame
    # Scale up the skeleton
    scale_factor = 1  # Adjust this value to increase or decrease the scale
    
    # Calculate frame and keypoints centers
    frame_center = np.array(black_frame.shape[1::-1]) // 2
    keypoints_center = np.mean(keypoints[0], axis=0)

    # Scale and center keypoints
    scaled_keypoints = [
        tuple(map(int, frame_center + (kp - keypoints_center) * scale_factor))
        for kp in keypoints[0]
    ]

    # # Draw skeleton
    # for i, (start, end) in enumerate(skeleton):
    #     if all(kp > 0 for kp in scaled_keypoints[start-1]) and all(kp > 0 for kp in scaled_keypoints[end-1]):
    #         start_point = tuple(map(int, scaled_keypoints[start-1]))
    #         end_point = tuple(map(int, scaled_keypoints[end-1]))
    #         cv2.line(black_frame, start_point, end_point, joint_colors[i % len(joint_colors)], 2)

    # # Draw and label keypoints
    # keypoint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]
    # for i, (x, y) in enumerate(scaled_keypoints):
    #     cv2.circle(black_frame, (x, y), 5, (255, 255, 255), -1)
    #     cv2.circle(black_frame, (x, y), 2, (0, 0, 0), -1)
    #     cv2.putText(black_frame, keypoint_labels[i], (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # # Show the black frame with centered skeleton
    # cv2.namedWindow('Skeleton Visualization', cv2.WINDOW_NORMAL)
    # cv2.imshow('Skeleton Visualization', black_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    for j, (x, y) in enumerate(keypoints[0]):
        cv2.circle(img, (int(x), int(y)), 3, joint_colors[j], -1)
        # cv2.putText(img, str(j+1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # # Visualize the frame
    # cv2.imshow('Frame', img)
    # cv2.waitKey(0)  # Wait for any key press
    # cv2.destroyAllWindows()  # Close the window after key press
 
    
    for i, (start, end) in enumerate(skeleton):
        if all(kp > 0 for kp in keypoints[0][start-1]) and all(kp > 0 for kp in keypoints[0][end-1]):
            start_point = tuple(map(int, keypoints[0][start-1]))
            end_point = tuple(map(int, keypoints[0][end-1]))
            cv2.line(img, start_point, end_point, joint_colors[i % len(joint_colors)], 2)

    # cv2.imshow('Frame', img)
    # cv2.waitKey(0)  # Wait for any key press
    # cv2.destroyAllWindows()  # Close the window after key press
