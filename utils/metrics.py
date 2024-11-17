import torch
import numpy as np

def calculate_iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou

def calculate_mAP(true_boxes, pred_boxes, true_labels, iou_threshold=0.5):
    # Calculate mean Average Precision
    ap_sum = 0
    num_classes = len(set(true_labels))
    
    for class_id in range(num_classes):
        true_positives = []
        false_positives = []
        
        for i in range(len(true_boxes)):
            if true_labels[i] == class_id:
                iou = calculate_iou(true_boxes[i], pred_boxes[i])
                if iou >= iou_threshold:
                    true_positives.append(1)
                    false_positives.append(0)
                else:
                    true_positives.append(0)
                    false_positives.append(1)
        
        true_positives = np.array(true_positives)
        false_positives = np.array(false_positives)
        
        # Calculate precision and recall
        cumsum = np.cumsum(true_positives)
        precision = cumsum / (np.arange(len(cumsum)) + 1)
        recall = cumsum / (np.sum(true_labels == class_id) + 1e-6)
        
        # Calculate average precision
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        ap_sum += ap
    
    mAP = ap_sum / num_classes
    return mAP

def calculate_keypoint_mse(true_keypoints, pred_keypoints):
    # Ensure inputs are tensors and have the same shape
    if not isinstance(true_keypoints, torch.Tensor):
        true_keypoints = torch.stack(true_keypoints)
    if not isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = torch.stack(pred_keypoints)
    
    # Ensure both tensors are on the same device
    device = true_keypoints.device
    pred_keypoints = pred_keypoints.to(device)
    
    # Calculate Mean Squared Error for keypoints
    mse = torch.mean((true_keypoints - pred_keypoints) ** 2)
    return mse.item()

def object_keypoint_similarity(pred_keypoints, true_keypoints, sigma=0.1):
        """
        Calculate Object Keypoint Similarity (OKS)
        
        Args:
        pred_keypoints (torch.Tensor): Predicted keypoints (N, K, 2)
        true_keypoints (torch.Tensor): Ground truth keypoints (N, K, 2)
        sigma (float): Scale factor
        
        Returns:
        torch.Tensor: OKS score
        """
        d = torch.sum((pred_keypoints - true_keypoints)**2, dim=2)
        s = torch.sum(torch.sum(true_keypoints[..., :2]**2, dim=2), dim=1).sqrt()
        oks = torch.exp(-d / (2 * (s[:, None] * sigma)**2 + 1e-9))
        return oks.mean()