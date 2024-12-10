import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import numpy as np
import json
import random
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

def draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2):
    x, y, w, h = [int(coord) for coord in bbox]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

def draw_keypoints_and_skeleton(img, keypoints):
    """
    Draws keypoints and skeleton on an image.
    Parameters:
    img (numpy.ndarray): The image on which to draw the keypoints and skeleton.
    keypoints (numpy.ndarray): An array of shape (N, 3) where N is the number of keypoints.
                               Each keypoint is represented by (x, y, v) where (x, y) are the coordinates
                               and v is the visibility flag (0: not visible, 1: visible).
    Returns:
    None: The function modifies the input image in place by drawing the keypoints and skeleton.
    """
    
    joint_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            (64, 64, 0), (64, 0, 64), (0, 64, 64)
        ]
    skeleton = [(1, 2), (1, 3), (1, 18), (2, 4), (3, 5), (6, 8), (6, 12), (6, 18), (7, 9), (7, 13), (7, 18), (8, 10), (9, 11), (12, 14), (12, 13), (13, 15), (14, 16), (15, 17)]

    for j, (x, y, v) in enumerate(keypoints):
        if v > 0:
            cv2.circle(img, (int(x), int(y)), 3, joint_colors[j], -1)
            cv2.putText(img, str(j+1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for i, (start_point, end_point) in enumerate(skeleton):
        if keypoints[start_point-1, 2] > 0 and keypoints[end_point-1, 2] > 0:
            start = tuple(keypoints[start_point-1, :2].astype(int))
            end = tuple(keypoints[end_point-1, :2].astype(int))
            cv2.line(img, start, end, joint_colors[i % len(joint_colors)], 2)

def process_image(img_info, annotation, base_path, img_width, img_height):
    """
    Processes an image by reading it from the specified path, converting its color space,
    and optionally drawing bounding boxes and keypoints based on the provided annotation.
    Args:
        img_info (dict): Dictionary containing image metadata, including 'path', 'width', and 'height'.
        annotation (dict): Dictionary containing annotation data, including 'bbox' and 'keypoints'.
                           If None, no annotations will be drawn.
        base_path (str): Base path to the directory containing the image.
        img_width (int): The width to which the image should be scaled.
        img_height (int): The height to which the image should be scaled.
    Returns:
        numpy.ndarray: The processed image with optional annotations drawn.
    """
    
    img_path = f"{base_path}/{img_info['path'].lstrip('../')}"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if annotation:
        bbox = [int(coord * img_width / img_info['width']) for coord in annotation['bbox']]
        draw_bounding_box(img, bbox)
        
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)
        keypoints[:, 0] = keypoints[:, 0] * img_width / img_info['width']
        keypoints[:, 1] = keypoints[:, 1] * img_height / img_info['height']
        
        draw_keypoints_and_skeleton(img, keypoints)
    
    return img

def visualise_dataset(json_file, base_path, num_images=2):
    """
    Visualises a subset of images from a dataset described in a JSON file.
    Parameters:
    json_file (str): Path to the JSON file containing dataset information.
    base_path (str): Base path to the directory containing the images.
    num_images (int, optional): Number of images to visualise. Defaults to 2.
    Returns:
    matplotlib.figure.Figure: A matplotlib figure object containing the visualised images.
    The JSON file should contain:
    - 'images': A list of dictionaries, each with keys 'id', 'file_name', 'width', and 'height'.
    - 'annotations': A list of dictionaries, each with keys 'image_id' and other annotation details.
    The function randomly selects `num_images` from the dataset, processes them, and displays them in a matplotlib figure.
    """
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    selected_images = random.sample(data['images'], num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(20, 6))
    
    for i, (ax, img_info) in enumerate(zip(axes, selected_images)):
        annotation = next((ann for ann in data['annotations'] if ann['image_id'] == img_info['id']), None)
        img = process_image(img_info, annotation, base_path, img_info['width'], img_info['height'])
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Image {i+1}: {img_info['file_name']}")
    
    plt.tight_layout()
    return fig


import numpy as np
import cv2

import cv2
import numpy as np

def visualise_dataloader_sample(ax, image, bbox, keypoints, label, title):
    """
    Visualize a sample from the dataloader by displaying the image with bounding boxes and keypoints.
    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot the image.
        image (torch.Tensor): The image tensor to be visualized.
        bbox (list or torch.Tensor): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        keypoints (torch.Tensor): The keypoints tensor with shape (num_keypoints, 3).
        label (int): The label index corresponding to the action class.
        title (str): The title for the plot.
    Returns:
        None
    """
    
    # Convert tensor image to numpy array and denormalize
    img_np = image.cpu().permute(1, 2, 0).detach().numpy()  # Ensure it's on CPU and detached
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Denormalize the image
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)  # Clip values to ensure they are within [0, 1]
    
    # Convert to uint8 and BGR format for OpenCV
    img_np = (img_np * 255).astype(np.uint8)  # Scale to [0, 255]
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    
    # Get image dimensions
    height, width = 1,1 #img_rgb.shape[:2]
    
    # Draw bounding box (converting from normalized to pixel space)
    # Expecting bbox in format [x_min, y_min, x_max, y_max]
    x_min = int(bbox[0] * width)
    y_min = int(bbox[1] * height)
    x_max = int(bbox[2] * width)
    y_max = int(bbox[3] * height)

    # Draw the bounding box on the image
    cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)  # Green box

    # Draw keypoints
    keypoints_pixel = keypoints.cpu().detach().view(-1, 3).numpy()  # Ensure it's on CPU and detached
    keypoints_pixel[:, 0] *= width  # Scale x to image width
    keypoints_pixel[:, 1] *= height  # Scale y to image height
    
    # Assuming draw_keypoints_and_skeleton is defined and works properly
    draw_keypoints_and_skeleton(img_rgb, keypoints_pixel)  

    # Display the image
    ax.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}: {['backhand', 'forehand', 'serve', 'ready_position'][label]}")
    ax.axis('off')


def visualize_results(all_images, all_labels, all_predictions, all_bboxes, all_keypoints, num_samples=3):
    """
    Visualize the results of image predictions by displaying the correct and predicted labels side by side.
    Parameters:
    - all_images (list or tensor): List or tensor of images.
    - all_labels (list or tensor): List or tensor of true labels.
    - all_predictions (list or tensor): List or tensor of predicted labels.
    - all_bboxes (list or tensor): List or tensor of bounding boxes.
    - all_keypoints (list or tensor): List or tensor of keypoints.
    - num_samples (int, optional): Number of samples to visualize. Default is 3.
    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib figure object containing the visualizations.
    The function randomly selects a specified number of samples from the provided data and visualizes them.
    Each sample is displayed in two columns: one for the correct label and one for the predicted label.
    Bounding boxes and keypoints are also displayed on the images.
    """
    
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 6*num_samples))
    fig.suptitle('Correct vs Predicted Results', fontsize=16)

    random_indices = random.sample(range(len(all_images)), num_samples)

    for i, idx in enumerate(random_indices):
        image = all_images[idx]
        true_label = all_labels[idx]
        pred_label = all_predictions[idx]
        bbox = all_bboxes[idx]
        keypoints = all_keypoints[idx].reshape(-1, 2)
        
        # Denormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.clamp(0, 1)  # Ensure values are in [0, 1] range
        
        # Display correct label image
        visualise_dataloader_sample(axs[i, 0], image, bbox, keypoints, true_label, 'Correct')
        
        # Display predicted label image
        visualise_dataloader_sample(axs[i, 1], image, bbox, keypoints, pred_label, 'Predicted')

        # Print bbox and keypoints values (in pixel coordinates)
        print(f"Sample {i+1}:")
        height, width = 224, 224  # Assuming resized images
        bbox_np = bbox.cpu().numpy()
        keypoints_np = keypoints.cpu().numpy()
        
        print("Bounding Box:")
        print(f"{'':>10}{'Correct':>15}{'Predicted':>15}")
        print(f"{'x:':<10}{bbox_np[0]*width:15.2f}{bbox_np[0]*width:15.2f}")
        print(f"{'y:':<10}{bbox_np[1]*height:15.2f}{bbox_np[1]*height:15.2f}")
        print(f"{'width:':<10}{bbox_np[2]*width:15.2f}{bbox_np[2]*width:15.2f}")
        print(f"{'height:':<10}{bbox_np[3]*height:15.2f}{bbox_np[3]*height:15.2f}")
        
        print("\nKeypoints:")
        print(f"{'Point':>5}{'Correct X':>15}{'Correct Y':>15}{'Predicted X':>15}{'Predicted Y':>15}")
        for j, (kx, ky) in enumerate(keypoints_np):
            print(f"{j+1:5d}{kx*width:15.2f}{ky*height:15.2f}{kx*width:15.2f}{ky*height:15.2f}")
        print()

    plt.tight_layout()
    return fig

def process_video(model, input_video_path, output_video_path, device, preprocess_image):
    """
    Processes an input video frame by frame, performs inference using a given model, and saves the annotated video.
    Args:
        model (torch.nn.Module): The model used for inference.
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output annotated video file.
        device (torch.device): The device (CPU or GPU) to run the model on.
        preprocess_image (function): A function to preprocess each frame before inference.
    Returns:
        None
    The function performs the following steps:
    1. Opens the input video file.
    2. Retrieves video properties such as width, height, frames per second (fps), and total frames.
    3. Creates a VideoWriter object to save the output video.
    4. Processes each frame of the video:
        - Converts the frame to a PIL Image.
        - Preprocesses the image and performs inference using the model.
        - Extracts keypoints, bounding box, and classification results from the model's output.
        - Draws the bounding box and keypoints on the frame.
        - Adds the predicted class label to the frame.
        - Writes the annotated frame to the output video.
    5. Releases video resources and saves the annotated video.
    """
    
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process each frame
    for _ in tqdm(range(min(100, total_frames)), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Perform inference
        preprocessed_image = preprocess_image(pil_image)
        with torch.no_grad():
            pred_keypoints, pred_bbox, pred_classification = model(preprocessed_image)
        
        # Convert the predicted keypoints and bounding box to numpy arrays
        keypoints = pred_keypoints.squeeze().cpu().numpy()
        bbox = pred_bbox.squeeze().cpu().numpy()

        # Convert the predicted class probabilities to numpy array and get the predicted class
        class_probs = F.softmax(pred_classification, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(class_probs)
        
        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x*width), int(y*height)), (int((x+w)*width), int((y+h)*height)), (0, 255, 0), 2)
        
        # Draw keypoints
        for kp in keypoints.reshape(-1, 2):
            cv2.circle(frame, (int(kp[0]*width), int(kp[1]*height)), 3, (255, 0, 0), -1)
        
        # Add class label
        class_name = f"Class {predicted_class}"
        cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write the frame
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video processing complete. Annotated video saved.")