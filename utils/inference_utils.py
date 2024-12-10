

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

def draw_keypoints_and_skeleton(img, keypoints,yolo):
    joint_colors = [[
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            (64, 64, 0), (64, 0, 64), (0, 64, 64)
        ],[
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            (64, 64, 0), (64, 0, 64), (0, 64, 64)
        ]]
    joint_names =[[
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
            "Right Knee", "Left Ankle", "Right Ankle", "Neck"
        ], [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
            "Right Knee", "Left Ankle", "Right Ankle"
        ]]
    skeleton = [
        [
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
        ],
        [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7]
        ]
    ]

    joint_colors = joint_colors[yolo]
    joint_names = joint_names[yolo]
    skeleton = skeleton[yolo]

    if keypoints:
        # Draw keypoints
        for i, (x, y, v) in enumerate(keypoints[0]):
            if v > 0:  # Only draw visible keypoints
                cv2.circle(img, (int(x), int(y)), 3, joint_colors[i % len(joint_colors)], -1)
                offset = 80 if yolo else 50
                if i % 2 == 0:
                    text_pos = (int(x) + offset, int(y) + offset//2)
                else:
                    text_pos = (int(x) - offset, int(y) - offset//2)
                cv2.line(img, (int(x), int(y)), text_pos, joint_colors[i % len(joint_colors)], 1)
                cv2.putText(img, joint_names[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, joint_colors[i % len(joint_colors)], 1, cv2.LINE_AA)

        # Draw skeleton
        for joint in skeleton:
            pt1 = keypoints[0][joint[0] - 1]
            pt2 = keypoints[0][joint[1] - 1]
            if pt1[2] > 0 and pt2[2] > 0:  # Only draw lines between visible keypoints
                color = joint_colors[joint[0] % len(joint_colors)]
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

def visualize_results(image, true_bbox, pred_bbox, true_keypoints, pred_keypoints,yolo=0):
    # Create a copy of the image for visualization
    original_image = image.copy()
    predicted_image = image.copy()

    # Draw true bounding box and keypoints on the original image
    cv2.rectangle(original_image, (int(true_bbox[0]), int(true_bbox[1])), (int(true_bbox[2]), int(true_bbox[3])), (0, 255, 0), 2)
    cv2.rectangle(original_image, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 0, 255), 2)
    
    draw_keypoints_and_skeleton(original_image, [true_keypoints],yolo)
    
    if yolo and pred_bbox:
        # Draw predicted bounding box and keypoints on the predicted image
        cv2.rectangle(predicted_image, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 0, 255), 2)
        draw_keypoints_and_skeleton(predicted_image, [pred_keypoints],yolo)
    else:
        # Draw predicted bounding box and keypoints on the predicted image
        cv2.rectangle(predicted_image, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[2]), int(pred_bbox[3])), (0, 0, 255), 2)
        draw_keypoints_and_skeleton(predicted_image, [pred_keypoints],yolo)

    # Concatenate the original and predicted images side by side
    comparison_image = np.concatenate((original_image, predicted_image), axis=1)

    # Display the comparison image
    cv2.imshow('Original vs Predicted', comparison_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()