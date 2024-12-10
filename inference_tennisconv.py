# import os
# import cv2
# import torch
# import json
# import argparse
# import os
# import cv2
# import torch
# import argparse
# import json
# from models.tennis_conv import TennisConv
# from utils.inference_utils import draw_keypoints_and_skeleton,get_player_direction

# def load_model(model_path):
#     model = TennisConv()
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.eval()
#     return model

# def preprocess_frame(frame):
#     # Resize the frame to the size expected by your model
#     target_size = (640, 640)  # Change this to your model's input size
#     frame_resized = cv2.resize(frame, target_size)

#     # Convert the frame to a float32 tensor and normalize pixel values
#     image_tensor = frame_resized.astype('float32') / 255.0  # Normalize to [0, 1]

#     # Convert to a PyTorch tensor and add batch dimension
#     image_tensor = torch.tensor(image_tensor).permute(2, 0, 1)  # Change from HWC to CHW format
#     image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

#     return image_tensor
# def process_media(input_path, output_dir, model, is_video=True):
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, os.path.basename(input_path))
#     data_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '_data.json')

#     inference_data = []

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     if is_video:
#         video = cv2.VideoCapture(input_path)
#         fps, width, height = [int(video.get(prop)) for prop in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]]
#         out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#         frame_count = 0

#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
            
#             # Preprocess the frame for your model
#             image_tensor = preprocess_frame(frame)  # Implement this function as needed
#             with torch.no_grad():
#                 results = model(image_tensor.to(device))

#             # Process results (assuming results are in a similar format to YOLO)
#             frame_data = {
#                 'frame_number': frame_count,
#                 'detections': []
#             }
            
#             for result in results:
#                 bbox = result.boxes.xyxy[0].tolist()
#                 keypoints = result.keypoints[0].xy.tolist() if result.keypoints else None

#                 # Draw keypoints or bounding boxes on the frame
#                 draw_keypoints_and_skeleton(frame, keypoints)  # Implement this function

#                 detection = {
#                     'id': result.boxes.id[0].item(),
#                     'confidence': result.boxes.conf[0].item(),
#                     'bbox': bbox,
#                     'keypoints': keypoints
#                 }
#                 frame_data['detections'].append(detection)

#             out.write(frame)  # Write the annotated frame to output
#             inference_data.append(frame_data)
#             frame_count += 1

#         video.release()
#         out.release()
#     else:
#         image = cv2.imread(input_path)
#         image_tensor = preprocess_frame(image)  # Implement this function
#         with torch.no_grad():
#             results = model(image_tensor.to(device))

#         for result in results:
#             draw_keypoints_and_skeleton(image, result.keypoints[0].xy.tolist())  # Implement this function

#         cv2.imwrite(output_path, image)

#         inference_data = [{
#             'detections': [{
#                 'confidence': result.boxes.conf[0].item(),
#                 'bbox': result.boxes.xyxy[0].tolist(),
#                 'keypoints': result.keypoints[0].xy.tolist() if result.keypoints else None
#             } for result in results]
#         }]

#     with open(data_output_path, 'w') as f:
#         json.dump(inference_data, f, indent=2)

#     print(f"Output {'video' if is_video else 'image'} saved to: {os.path.abspath(output_path)}")
#     print(f"Inference data saved to: {os.path.abspath(data_output_path)}")

# def main(input_path, output_dir):
#     model = load_model("checkpoints\\best_model.pth.tar")
#     is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))  # Check if input is a video
#     process_media(input_path, output_dir, model, is_video)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Inference script for TennisConv model")
#     parser.add_argument("input_path", help="Path to the input video or image file")
#     parser.add_argument("--output_dir", default="testing/runs_tennisconv1", help="Directory to save the output file")
#     args = parser.parse_args()

#     main(args.input_path, args.output_dir)
    

# # python .\inference_tennisconv.py testing/test3.mp4

import os
import cv2
import torch
import argparse
import json
from models.tennis_conv import SimpleTennisConv, TennisConvResidual, EnhancedTennisConv
from utils.inference_utils import draw_keypoints_and_skeleton, get_player_direction
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EnhancedTennisConv(backbone_name='efficientnet_b7').to(device)
checkpoint = torch.load("checkpointsEnhanced\\best_model.pth.tar",weights_only=True, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def preprocess_frame(frame, device):
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
    frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device)
    return frame

def postprocess_keypoints(keypoints, frame_shape):
    keypoints = keypoints.view(-1, 3).cpu().detach().numpy()
    keypoints[:, :2] *= frame_shape[:2]  # Scale keypoints to original frame size
    return [keypoints[:, :2].tolist()]

def postprocess_bboxes(bboxes, frame_shape):
    bboxes = bboxes.view(-1).cpu().detach().numpy()
    bboxes[:2] *= frame_shape[:2]  # Scale bbox coordinates to original frame size
    bboxes[2:] *= frame_shape[:2]
    return bboxes

def process_media(input_path, output_dir, is_video=True):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    data_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '_data.json')

    inference_data = []

    if is_video:
        video = cv2.VideoCapture(input_path)
        fps, width, height = [int(video.get(prop)) for prop in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0

        prev_positions = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            input_tensor = preprocess_frame(frame, device)
            keypoints, bboxes, classification_logits = model(input_tensor)
            
            keypoints = postprocess_keypoints(keypoints, frame.shape)
            bboxes = postprocess_bboxes(bboxes, frame.shape)
            classification = torch.argmax(classification_logits, dim=1).item()
            
            # print(f"Frame {frame_count}: {classification}")
            # print(f"{len(keypoints)} Keypoints:\n{keypoints}")
            # print(f"Bbox: {bboxes}")
            
            
            frame_data = {
                'frame_number': frame_count,
                'detections': []
            }
            
            if keypoints is not None and bboxes is not None:
                draw_keypoints_and_skeleton(frame, keypoints)

                current_position = [(bboxes[0] + bboxes[2]) / 2, (bboxes[1] + bboxes[3]) / 2]
                prev_positions.append(current_position)
                
                direction = get_player_direction(prev_positions, current_position, fps)
                
                cv2.putText(frame, f"Direction: {direction}", (int(width*0.7), int(height*0.85)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                detection = {
                    'id': frame_count,
                    'confidence': 1.0,  # Assuming confidence is always 1.0 for this model
                    'bbox': bboxes,
                    'keypoints': keypoints,
                    'direction': direction,
                    'classification': classification
                }
                frame_data['detections'].append(detection)
            
            out.write(frame)
            inference_data.append(frame_data)
            print(f"Frame {frame_count} processed")
            frame_count += 1

        video.release()
        out.release()
    else:
        image = cv2.imread(input_path)
        input_tensor = preprocess_frame(image, device)
        keypoints, bboxes, classification_logits = model(input_tensor)
        
        keypoints = postprocess_keypoints(keypoints, image.shape)
        bboxes = postprocess_bboxes(bboxes, image.shape)
        classification = torch.argmax(classification_logits, dim=1).item()
        
        draw_keypoints_and_skeleton(image, keypoints)
        cv2.putText(image, f"Class: {classification}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, image)

        inference_data = [{
            'detections': [{
                'confidence': 1.0,  # Assuming confidence is always 1.0 for this model
                'bbox': bboxes.tolist(),
                'keypoints': keypoints.tolist(),
                'classification': classification
            }]
        }]

    with open(data_output_path, 'w') as f:
        json.dump(inference_data, f, indent=2)

    print(f"Output {'video' if is_video else 'image'} saved to: {os.path.abspath(output_path)}")
    print(f"Inference data saved to: {os.path.abspath(data_output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files for pose estimation")
    parser.add_argument("input_path", help="Path to the input video or image file")
    parser.add_argument("--output_dir", default="testing/runs_tennisconv1", help="Directory to save the output file")
    parser.add_argument("--image", action="store_true", help="Process as image instead of video")
    args = parser.parse_args()

    process_media(args.input_path, args.output_dir, not args.image)

# Example usage:
# python .\inference_tennisconv.py testing/test2.mp4