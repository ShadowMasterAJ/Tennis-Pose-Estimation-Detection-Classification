import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error
from dataloader import get_dataloaders
from utils.inference_utils import visualize_results
from models.tennis_conv import SimpleTennisConv, TennisConvResidual, EnhancedTennisConv 
from models.tennis_brnn import TennisPoseSPP
from models.tennisnet import TennisNet
from training.config import Config

def calculate_iou(true_bbox, pred_bbox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters:
    true_bbox (list or tuple): The ground truth bounding box in the format [x1, y1, x2, y2],
                               where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    pred_bbox (list or tuple): The predicted bounding box in the format [x1, y1, x2, y2],
                               where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    Returns:
    float: The IoU of the two bounding boxes.
    """
    xA = max(true_bbox[0], pred_bbox[0])
    yA = max(true_bbox[1], pred_bbox[1])
    xB = min(true_bbox[2], pred_bbox[2])
    yB = min(true_bbox[3], pred_bbox[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    boxBArea = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def check_keypoint_alignment(true_keypoints, pred_keypoints):
    joint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
        "Right Knee", "Left Ankle", "Right Ankle"
    ]

    for i, (true_kp, pred_kp) in enumerate(zip(true_keypoints, pred_keypoints)):
        mse = mean_squared_error(true_kp[:2], pred_kp[:2])
        mse_values = [mean_squared_error(t[:2], p[:2]) for t, p in zip(true_keypoints, pred_keypoints)]
        mse_mean = np.mean(mse_values)
        mse_std = np.std(mse_values)
        mse_threshold = mse_mean + 4 * mse_std  # Outlier threshold based on mean and standard deviation

        if mse >= mse_threshold:
            print(f"\n{'*'*20}Potential misalignment detected for {joint_names[i]}{'*'*20}")
            print(f"{joint_names[i]}:\t True: {true_kp},\t Pred: {pred_kp},\t MSE: {mse}\n")

def denormalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

def evaluate_yolo_model(model_path, test_images_path, test_labels_path, device):
    model = YOLO(model_path).to(device)
    
    bbox_ious = []
    keypoint_errors = []
    correct_classifications = 0
    total_samples = 0

    for i, img_file in enumerate(os.listdir(test_images_path)):
        img_path = os.path.join(test_images_path, img_file)
        label_path = os.path.join(test_labels_path, os.path.splitext(img_file)[0] + '.txt')
        
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        results = model(image, device=device, conf = 0.01)
        
        with open(label_path, 'r') as f:
            label_data = f.read().strip().split()
            true_class = int(label_data[0])
            true_bbox = np.array(label_data[1:5], dtype=np.float32)
            true_keypoints = np.array(label_data[5:], dtype=np.float32).reshape(-1, 3)
        # Unnormalize true keypoints
        true_keypoints[:, 0] *= width
        true_keypoints[:, 1] *= height
        
        # Unnormalize true bounding box from (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
        true_bbox[0] *= width  # x_center
        true_bbox[1] *= height # y_center
        true_bbox[2] *= width  # width
        true_bbox[3] *= height # height
        
        true_bbox_converted = np.array([
            true_bbox[0] - true_bbox[2] / 2,  # x_min
            true_bbox[1] - true_bbox[3] / 2,  # y_min
            true_bbox[0] + true_bbox[2] / 2,  # x_max
            true_bbox[1] + true_bbox[3] / 2   # y_max
        ])
        
        if not results[0]:
            visualize_results(image, true_bbox_converted, None, true_keypoints, None, yolo=1)
        
        pred_bbox = results[0].boxes.xyxy[0].cpu().numpy()
        pred_keypoints = results[0].keypoints.data[0].cpu().numpy()
        pred_class = results[0].boxes.cls[0].item()
        
        check_keypoint_alignment(true_keypoints, pred_keypoints)
        
        # Calculate IoU for bounding boxes
        iou = calculate_iou(true_bbox_converted, pred_bbox)
        bbox_ious.append(iou)
        
        # Calculate MSE for keypoints
        mse = mean_squared_error(true_keypoints[:, :2], pred_keypoints[:, :2])
        keypoint_errors.append(mse)
        
        
        if pred_class == true_class:
            correct_classifications += 1
        total_samples += 1
        
        # # # Visualize the results for this sample
        # visualize_results(image, true_bbox_converted, pred_bbox, true_keypoints, pred_keypoints, yolo=1)

    mean_iou = np.mean(bbox_ious)
    mean_keypoint_error = np.mean(keypoint_errors)
    classification_accuracy = correct_classifications / total_samples

    print(f"\n{'*'*20} Evaluation Results {'*'*20}")
    print(f"Mean IoU: {mean_iou}")
    print(f"Mean Keypoint Error (MSE): {mean_keypoint_error}")
    print(f"Classification Accuracy: {classification_accuracy}")
    print(f"Total Samples: {total_samples}")
    print(f"{'*'*60}\n")
    
    return mean_iou, mean_keypoint_error, classification_accuracy
    
def evaluate_tennis_conv_model(model,model_weights_path,device,seq):
    model.to(device)
    model.eval()
    
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    json_files = os.listdir("og_dataset/annotations")
    base_path = "og_dataset"
    _, _, test_loader = get_dataloaders(json_files, base_path, batch_size=32)

    bbox_ious = []
    keypoint_errors = []
    correct_classifications = 0
    total_samples = 0

    for images, bboxes, keypoints, labels in tqdm(test_loader, desc="Evaluating"):
        images = images[:, -1, :, :, :].to(device)  # Only take the last frame
        bboxes = bboxes[:, -1, :].to(device)        # Only take the last frame
        keypoints = keypoints[:, -1, :].to(device)  # Only take the last frame
        labels = labels.to(device)           # Only take the last frame
        height, width = 720, 1280
        
        with torch.no_grad():
            pred_keypoints, pred_bboxes, pred_classes = model(images)
        
        pred_keypoints = pred_keypoints.cpu().numpy().reshape(-1, 18, 3)
        pred_bboxes = pred_bboxes.cpu().numpy().reshape(-1, 4)
        pred_classes = pred_classes.argmax(dim=1).cpu().numpy()
        
        for i in range(images.size(0)):
            true_keypoint = keypoints[i].cpu().numpy().reshape(-1, 3)
            true_bbox = bboxes[i].cpu().numpy().reshape(-1)
            true_class = labels[i].cpu().item()
            resized_image = cv2.resize(images[i].cpu().numpy().transpose(1, 2, 0), (1280, 720))

            pred_keypoint = pred_keypoints[i]
            pred_bbox = pred_bboxes[i]
            pred_class = pred_classes[i]
            
            pred_keypoint[:, 0] *= width  # x
            pred_keypoint[:, 1] *= height  # y

            # true_keypoints[:, 0] *= width
            # true_keypoints[:, 1] *= height
            
            pred_bbox_converted = np.array([
                pred_bbox[0] * width,  # x_min
                pred_bbox[1] * height,  # y_min
                pred_bbox[2] * width,  # x_max
                pred_bbox[3] * height   # y_max
            ])
            true_bbox_converted = np.array([
                true_bbox[0],  # x_min
                true_bbox[1],  # y_min
                true_bbox[0] + true_bbox[2],  # x_max
                true_bbox[1] + true_bbox[3]   # y_max
            ])
            
            # # # check_keypoint_alignment(true_keypoints, pred_keypoint)
            # visualize_results(resized_image, true_bbox_converted, pred_bbox, true_keypoints, pred_keypoint, yolo=0)
            
            # Calculate IoU for bounding boxes
            iou = calculate_iou(true_bbox_converted, pred_bbox_converted)
            bbox_ious.append(iou)
            
            # Calculate MSE for keypoints
            mse = mean_squared_error(true_keypoint[:, :2], pred_keypoint[:, :2])
            keypoint_errors.append(mse)
            
            if pred_class == true_class:
                correct_classifications += 1
            total_samples += 1
            
            if iou < 0.1:
                print('IOU:',iou)
                print(true_bbox_converted)
                print(pred_bbox_converted)
                visualize_results(resized_image, true_bbox_converted, pred_bbox_converted, true_keypoint, pred_keypoint, yolo=0)
            # if mse > 50:
            #     print('MSE:',mse)
            #     print(true_keypoints)
            #     print(pred_keypoint)
            #     visualize_results(resized_image, true_bbox_converted, pred_bbox_converted, true_keypoints, pred_keypoint, yolo=0)

    mean_iou = np.mean(bbox_ious)
    mean_keypoint_error = np.mean(keypoint_errors)
    classification_accuracy = correct_classifications / total_samples

    print(f"\n{'*'*20} Evaluation Results {'*'*20}")
    print(f"Mean IoU: {mean_iou}")
    print(f"Mean Keypoint Error (MSE): {mean_keypoint_error}")
    print(f"Classification Accuracy: {classification_accuracy}")
    print(f"Total Samples: {total_samples}")
    print(f"{'*'*60}\n")
    
    return mean_iou, mean_keypoint_error, classification_accuracy

def evaluate_tennis_brnn_model(model, model_weights_path, device):
    model.to(device)
    model.eval()
    
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    json_files = os.listdir("og_dataset/annotations")
    base_path = "og_dataset"
    _, _, test_loader = get_dataloaders(json_files, base_path, batch_size=Config.BATCH_SIZE,sequence_length=Config.SEQ_LENGTH)

    bbox_ious = []
    keypoint_errors = []
    correct_classifications = 0
    total_samples = 0

    for sequences, bboxes, keypoints, labels in tqdm(test_loader, desc="Evaluating"):
        sequences = sequences.to(device)  # Batch of sequences
        bboxes = bboxes.to(device)
        keypoints = keypoints.to(device)
        labels = labels.to(device)
        height, width = 720, 1280
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        with torch.no_grad():
            pred_keypoints, pred_bboxes, pred_classes = model(sequences)
        
        pred_keypoints = pred_keypoints.cpu().numpy().reshape(-1, 18, 3)
        pred_bboxes = pred_bboxes.cpu().numpy().reshape(-1, 4)
        pred_classes = pred_classes.argmax(dim=1).cpu().numpy()
        
        for i in range(sequences.size(0)):
            true_keypoint = keypoints[i,-1].cpu().numpy().reshape(-1, 3)
            true_bbox = bboxes[i,-1].cpu().numpy().reshape(-1)
            true_class = labels[i].cpu().item()
            resized_image = denormalize(sequences[i, -1].cpu(), mean, std)

            pred_keypoint = pred_keypoints[i]
            pred_bbox = pred_bboxes[i]
            pred_class = pred_classes[i]
            
            true_bbox_converted = np.array([
                true_bbox[0],  # x_min
                true_bbox[1],  # y_min
                true_bbox[2],  # x_max
                true_bbox[3]   # y_max
            ])
            
            # Calculate IoU for bounding boxes
            iou = calculate_iou(true_bbox_converted, pred_bbox)
            bbox_ious.append(iou)
            
            # Calculate MSE for keypoints
            mse = mean_squared_error(true_keypoint[:, :2], pred_keypoint[:, :2])
            keypoint_errors.append(mse)
            
            if pred_class == true_class:
                correct_classifications += 1
            total_samples += 1
            
            # if iou < 0.1:
            #     print('IOU:', iou)
            #     print(true_bbox_converted)
            #     print(pred_bbox)
            #     visualize_results(resized_image, true_bbox_converted, pred_bbox, true_keypoint, pred_keypoint, yolo=0)

    mean_iou = np.mean(bbox_ious)
    mean_keypoint_error = np.mean(keypoint_errors)
    classification_accuracy = correct_classifications / total_samples

    print(f"\n{'*'*20} Evaluation Results {'*'*20}")
    print(f"Mean IoU: {mean_iou}")
    print(f"Mean Keypoint Error (MSE): {mean_keypoint_error}")
    print(f"Classification Accuracy: {classification_accuracy}")
    print(f"Total Samples: {total_samples}")
    print(f"{'*'*60}\n")
    
    return mean_iou, mean_keypoint_error, classification_accuracy
                                    
if __name__ == '__main__':
    models = os.listdir("tennis_pose_estimation")
    test_images_path = "dataset_yolo/images/test"
    test_labels_path = "dataset_yolo/labels/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights_map = {
        # "TennisNet": "checkpoints_net\\best_model.pth.tar",
        "TennisPoseSPP": "checkpoints_spp\\best_model.pth.tar",
    }
    
    test_images_path = "dataset_yolo/images/test"
    test_labels_path = "dataset_yolo/labels/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # for model_name, weights_path in model_weights_map.items():
    #     print(f"Evaluating {model_name}...")
    #     model_class = globals()[model_name]
    #     model = model_class()
    #     mean_iou, mean_keypoint_error, classification_accuracy = evaluate_tennis_brnn_model(model, weights_path, device)
    #     with open("custom_eval_results.txt", "a") as f:
    #         f.write(f"\n{'*'*20} Evaluation Results {'*'*20}\n")
    #         f.write(f"Model: {model_name}\n")
    #         f.write(f"{'-'*60}\n")
    #         f.write(f"Mean BBox IoU: {mean_iou}\n")
    #         f.write(f"Mean Keypoint Error (MSE): {mean_keypoint_error}\n")
    #         f.write(f"Classification Accuracy: {classification_accuracy}\n")
    #         f.write(f"{'*'*60}\n")
            
    # for model in models:
    #     model_path = f"tennis_pose_estimation\\{model}\\weights\\best.pt"
    #     mean_iou, mean_keypoint_error, classification_accuracy = evaluate_yolo_model(model_path, test_images_path, test_labels_path, device)        
    #     with open("tennis_pose_estimation/eval_results.txt", "a") as f:
    #         f.write(f"\n{'*'*20} Evaluation Results {'*'*20}\n")
    #         f.write(f"Model: {model}\n")
    #         f.write(f"{'-'*60}\n")
    #         f.write(f"Mean BBox IoU: {mean_iou}\n")
    #         f.write(f"Mean Keypoint Error (MSE): {mean_keypoint_error}\n")
    #         f.write(f"Classification Accuracy: {classification_accuracy}\n")
    #         f.write(f"{'*'*60}\n")
    
    for model, path in model_weights_map:

        mean_iou, mean_keypoint_error, classification_accuracy = evaluate_yolo_model(path, test_images_path, test_labels_path, device)        
        with open("tennis_pose_estimation/eval_results.txt", "a") as f:
            f.write(f"\n{'*'*20} Evaluation Results {'*'*20}\n")
            f.write(f"Model: {model}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"Mean BBox IoU: {mean_iou}\n")
            f.write(f"Mean Keypoint Error (MSE): {mean_keypoint_error}\n")
            f.write(f"Classification Accuracy: {classification_accuracy}\n")
            f.write(f"{'*'*60}\n")

