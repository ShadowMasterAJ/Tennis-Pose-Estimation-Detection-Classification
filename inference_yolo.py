import os
import cv2
import torch
import argparse
import json
from ultralytics import YOLO
from utils.inference_utils import draw_keypoints_and_skeleton, get_player_direction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model = YOLO("tennis_pose_estimation\\tennis_pose_estimation_continued\weights\\best.pt").to(device)
model = YOLO("tennis_pose_estimation\\tennis_pose_estimation_nano2\weights\\best.pt").to(device)


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
            results = model.track(frame, persist=True, device=device)
            filtered_results = [r for r in results if r.boxes.id is not None and r.boxes.id[0].item() <= 10 and r.boxes.conf[0].item() > 0.4]
            
            frame_data = {
                'frame_number': frame_count,
                'detections': []
            }
            
            if filtered_results:
                result = filtered_results[0]
                bbox = result.boxes.xyxy[0].tolist()
                keypoints = result.keypoints[0].xy.tolist() if result.keypoints else None
                
                if keypoints:
                    draw_keypoints_and_skeleton(frame, keypoints)

                current_position = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                prev_positions.append(current_position)
                
                
                direction = get_player_direction(prev_positions, current_position,fps)
                
                annotated_frame = result.plot(kpt_radius=0,font_size=15,line_width=2)
                 
                cv2.putText(annotated_frame, f"Direction: {direction}", (int(width*0.7), int(height*0.85)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2),
                
                detection = {
                    'id': result.boxes.id[0].item(),
                    'confidence': result.boxes.conf[0].item(),
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'direction': direction
                }
                frame_data['detections'].append(detection)
            else:
                annotated_frame = frame
            
            out.write(annotated_frame)
            inference_data.append(frame_data)
            frame_count += 1

        video.release()
        out.release()
    else:
        image = cv2.imread(input_path)
        results = model(image, device=device)
        for result in results:
            draw_keypoints_and_skeleton(image,result.keypoints[0].xy.tolist())
        annotated_image = results[0].plot(kpt_radius=2)

        cv2.imwrite(output_path, annotated_image)

        inference_data = [{
            'detections': [{
                'confidence': result.boxes.conf[0].item(),
                'bbox': result.boxes.xyxy[0].tolist(),
                'keypoints': result.keypoints[0].xy.tolist() if result.keypoints else None
            } for result in results]
        }]

    with open(data_output_path, 'w') as f:
        json.dump(inference_data, f, indent=2)

    print(f"Output {'video' if is_video else 'image'} saved to: {os.path.abspath(output_path)}")
    print(f"Inference data saved to: {os.path.abspath(data_output_path)}")

def compare_models(input_path, output_dir, model1, model2, is_video=True):
    """
    Compare two models by running inference on the same input and displaying results side by side.
    
    Args:
    input_path (str): Path to the input video or image file
    output_dir (str): Directory to save the output file
    model1 (YOLO): First YOLO model
    model2 (YOLO): Second YOLO model
    is_video (bool): True if processing a video, False for image
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_comparison{'_video' if is_video else '_image'}.mp4")
    data_output_path = os.path.join(output_dir, f"{base_name}_comparison_data.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process media for both models
    inference_data1 = process_media(input_path, output_dir, is_video, model1, suffix="_model1")
    inference_data2 = process_media(input_path, output_dir, is_video, model2, suffix="_model2")
    
    if is_video:
        video = cv2.VideoCapture(input_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*2, height))
        
        for frame_count in range(total_frames):
            frame1 = cv2.imread(os.path.join(output_dir, f"{base_name}_model1_frame_{frame_count}.jpg"))
            frame2 = cv2.imread(os.path.join(output_dir, f"{base_name}_model2_frame_{frame_count}.jpg"))
            
            combined_frame = np.hstack((frame1, frame2))
            
            cv2.putText(combined_frame, "Model 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, "Model 2", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(combined_frame)
            
            print(f"\rProcessing frame {frame_count+1}/{total_frames}", end="")
        
        out.release()
        print("\nVideo processing completed.")
    else:
        frame1 = cv2.imread(os.path.join(output_dir, f"{base_name}_model1.jpg"))
        frame2 = cv2.imread(os.path.join(output_dir, f"{base_name}_model2.jpg"))
        
        combined_image = np.hstack((frame1, frame2))
        
        cv2.putText(combined_image, "Model 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, "Model 2", (frame1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, combined_image)
        print("Image processing completed.")

    # Combine inference data
    combined_inference_data = []
    for data1, data2 in zip(inference_data1, inference_data2):
        combined_data = {
            'frame': data1['frame'],
            'model1_detections': data1['detections'],
            'model2_detections': data2['detections']
        }
        combined_inference_data.append(combined_data)

    with open(data_output_path, 'w') as f:
        json.dump(combined_inference_data, f, indent=2)

    print(f"Output {'video' if is_video else 'image'} saved to: {os.path.abspath(output_path)}")
    print(f"Inference data saved to: {os.path.abspath(data_output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process media files for pose estimation")
    parser.add_argument("input_path", help="Path to the input video or image file")
    parser.add_argument("--output_dir", default="testing/runs_nano", help="Directory to save the output file")
    parser.add_argument("--image", action="store_true", help="Process as image instead of video")
    args = parser.parse_args()

    process_media(args.input_path, args.output_dir, not args.image)

# Example usage:
# python inference.py testing/test3.mp4
# python inference.py testing/test.png testing --image
