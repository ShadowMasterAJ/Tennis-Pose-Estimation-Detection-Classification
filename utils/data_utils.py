from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class TennisDataset(Dataset):
    def __init__(self, data, transform=None, sequence_length=16):
        """
        Args:
            data (list): A list of dictionaries with keys 'image_path', 'bbox', 'keypoints', 'label'.
            transform (callable, optional): Optional transform to be applied to each frame.
            sequence_length (int): Number of frames in each sequence.
        """
        self.data = data
        self.transform = transform
        self.sequence_length = sequence_length
        self.grouped_data = self._group_sequences()

    def _group_sequences(self):
        """
        Group frames into sequences based on their label (e.g., video ID or shot type).
        Assumes data is sorted by some temporal property (e.g., video frame order).
        """
        grouped = {}
        for item in self.data:
            label = item['label']  # You can also use a unique 'video_id' if available
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(item)
        
        sequences = []
        for label, frames in grouped.items():
            for i in range(len(frames) - self.sequence_length + 1):
                sequences.append(frames[i:i + self.sequence_length])
        
        return sequences

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        """
        Returns:
            frames (Tensor): A sequence of frames with shape (sequence_length, C, H, W).
            bboxes (Tensor): Corresponding bounding boxes for the sequence (sequence_length, 4).
            keypoints (Tensor): Corresponding keypoints for the sequence (sequence_length, num_keypoints * 3).
            label (int): Shot type label for the sequence.
        """
        sequence = self.grouped_data[idx]
        
        frames = []
        bboxes = []
        keypoints = []
        labels = []
        
        for item in sequence:
            image = Image.open(item['image_path']).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
            bboxes.append(torch.tensor(item['bbox'], dtype=torch.float32))
            keypoints.append(torch.tensor(item['keypoints'], dtype=torch.float32))
            labels.append(item['label'])

        # Stack tensors
        frames = torch.stack(frames)  # (sequence_length, C, H, W)
        bboxes = torch.stack(bboxes)  # (sequence_length, 4)
        keypoints = torch.stack(keypoints)  # (sequence_length, num_keypoints * 3)
        label = torch.tensor(['backhand', 'forehand', 'serve', 'ready_position'].index(sequence[0]['label']))  # Assuming the label is the same for the entire sequence

        return frames, bboxes, keypoints, label

def get_transform(train: bool = True) -> transforms.Compose:
    """ Define transformations for training and validation datasets. """
    if train:
        return transforms.Compose([
            transforms.Resize((320, 320)),  # Resize images to 640x640
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness/contrast/saturation/hue
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian Blur
            transforms.ToTensor(),  # Convert PIL Image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((640, 640)),  # Resize images to 640x640
            transforms.ToTensor(),  # Convert PIL Image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

def normalize_bbox(bbox, width, height):
    """ Normalize and clip the bounding box to [0, 1]. """
    x_min, y_min, w, h = bbox
    return [
        max(0, min(1, x_min / width)),
        max(0, min(1, y_min / height)),
        max(0, min(1, (x_min + w) / width)),  # Normalize x_max
        max(0, min(1, (y_min + h) / height))  # Normalize y_max
    ]

def normalize_keypoints(keypoints, width, height):
    """ Normalize the keypoints based on image width and height. """
    normalized = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i+3]
        normalized.extend([
            max(0, min(1, x / width)),  # Normalize x
            max(0, min(1, y / height)),  # Normalize y
            v  # Keep visibility as is
        ])
    return normalized


def create_yolo_dataset(json_files, base_path, yolo_dataset_path):
    os.makedirs(yolo_dataset_path, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_dataset_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset_path, 'labels', split), exist_ok=True)

    all_data = []
    class_names = []

    for class_id, json_file in enumerate(json_files):
        class_name = json_file.split('.')[0]
        class_names.append(class_name)
        
        with open(os.path.join(base_path, 'annotations', json_file), 'r') as f:
            data = json.load(f)
        
        for img_info in data['images']:
            img_path = os.path.join(base_path, img_info['path'].lstrip('../'))
            img_width, img_height = img_info['width'], img_info['height']
            
            annotation = next((ann for ann in data['annotations'] if ann['image_id'] == img_info['id']), None)
            
            if annotation:
                bbox = normalize_bbox(annotation['bbox'], img_width, img_height)
                keypoints = normalize_keypoints(annotation['keypoints'], img_width, img_height)
                
                all_data.append({
                    'image_path': img_path,
                    'class_id': class_id,
                    'bbox': bbox,
                    'keypoints': keypoints
                })

    train_data, val_test_data = train_test_split(all_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    def process_split(split_name, data):
        for item in data:
            dst_img_path = os.path.join(yolo_dataset_path, 'images', split_name, os.path.basename(item['image_path']))
            shutil.copy(item['image_path'], dst_img_path)

            label_content = f"{item['class_id']} {' '.join(map(str, item['bbox'])) } {' '.join(map(str, item['keypoints']))}"
            label_filename = os.path.splitext(os.path.basename(item['image_path']))[0] + '.txt'
            with open(os.path.join(yolo_dataset_path, 'labels', split_name, label_filename), 'w') as f:
                f.write(label_content)

    process_split('train', train_data)
    process_split('val', val_data)
    process_split('test', test_data)

    yaml_content = f"""
path: {os.path.abspath(yolo_dataset_path)}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}

# Keypoint information
kpt_shape: [18, 3]  # number of keypoints, number of dimensions (x, y, visibility)
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]
"""
    with open(os.path.join(yolo_dataset_path, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"YOLO dataset created at {yolo_dataset_path}")
