import os
import json
import random
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class TennisDataset(Dataset):
    def __init__(self, data: List[Dict], transform=None, sequence_length: Optional[int] = None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.sequence_length:
            sequence = self.data[idx]
            frames, bboxes, keypoints = [], [], []
            for item in sequence:
                original_width, original_height = item['width'], item['height']

                image = Image.open(item['image_path']).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                
                new_height, new_width = image.shape[1], image.shape[2]

                # Compute scaling factors
                scale_x = new_width / original_width
                scale_y = new_height / original_height

                frames.append(image)
                bboxes.append(normalize_bbox(item['bbox'], scale_x, scale_y))
                keypoints.append(normalize_keypoints(item['keypoints'], scale_x, scale_y))
            frames = torch.stack(frames)
            bboxes = torch.stack(bboxes)
            keypoints = torch.stack(keypoints)
            label = torch.tensor(['backhand', 'forehand', 'serve', 'ready_position'].index(sequence[0]['label']))
            return frames, bboxes, keypoints, label
        
        else:
            item = self.data[idx]
            image = Image.open(item['image_path']).convert("RGB")
            original_width, original_height = image.size

            if self.transform:
                image = self.transform(image)
                
            new_height, new_width = image.shape[1], image.shape[2]

            # Compute scaling factors
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            bboxes = normalize_bbox(item['bbox'], scale_x, scale_y)
            keypoints = normalize_keypoints(item['keypoints'], scale_x, scale_y)
            label = torch.tensor(['backhand', 'forehand', 'serve', 'ready_position'].index(item['label']))
        
            return image, bboxes, keypoints, label

def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
def get_val_transform():
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def normalize_bbox(bbox: List[float], scale_x: float, scale_y: float) -> torch.Tensor:
    x, y, width, height = bbox
    xmin = max(0, x * scale_x)
    ymin = max(0, y * scale_y)
    xmax = max(0, (x + width) * scale_x)
    ymax = max(0, (y + height) * scale_y)
    normalized = [xmin, ymin, xmax, ymax]
    return torch.tensor(normalized, dtype=torch.float32)

def normalize_keypoints(keypoints: List[float], scale_x: float, scale_y: float) -> torch.Tensor:
    normalized = []
    for i in range(0, len(keypoints), 3):
        x = max(0, keypoints[i] * scale_x)
        y = max(0, keypoints[i + 1] * scale_y)
        v = keypoints[i + 2]
        normalized.extend([x, y, v])
    return torch.tensor(normalized, dtype=torch.float32)

def load_annotations(json_files: List[str], base_path: str) -> List[Dict]:
    all_data = []
    for json_file in json_files:
        json_file_path = os.path.join(base_path, 'annotations', json_file)
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Annotation file {json_file_path} not found.")
            continue

        shot_type = data['categories'][0]['name'].lower()
        for img_info in data['images']:
            img_path = os.path.join(base_path, img_info['path'].lstrip('../'))
            annotation = next((ann for ann in data['annotations'] if ann['image_id'] == img_info['id']), None)
            if annotation:
                all_data.append({
                    'image_path': img_path,
                    'bbox': annotation['bbox'],
                    'keypoints': annotation['keypoints'],
                    'label': shot_type,
                    'id': img_info['id'],
                    'height': img_info['height'],
                    'width': img_info['width']
                })
    return all_data

def sequentialize_data(data: List[Dict], sequence_length: int) -> List[List[Dict]]:
    sequences = []
    data.sort(key=lambda x: x['id'])
    for i in range(0, len(data), 500):
        batch = data[i:i + 500]
        for j in range(len(batch) - sequence_length + 1):
            sequence = batch[j:j + sequence_length]
            sequences.append(sequence)
    
    return sequences

def split_data(data: List[Dict], sequence_length: int = 1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if sequence_length > 1:
        # Split data sequentially
        
        train_ratio = 0.7
        val_ratio = 0.15

        total_frames = len(data)
        total_categories = 4

        # Calculate sizes
        train_size = int(train_ratio * total_frames)
        val_size = int(val_ratio * total_frames)

        # Initialize splits
        train_data = []
        val_data = []
        test_data = []
        
        category_data = [[] for _ in range(total_categories)]
        for i, ls in enumerate(category_data):
            ls.append(data[i*(len(data)//total_categories):(i+1)*(len(data)//total_categories)])
        
        category_data = [item for sublist in category_data for item in sublist]
        # Function to round-robin select frames
        def round_robin_select(target_size, start_indices):
            selected_data = []
            indices = start_indices.copy()
            
            while len(selected_data) < target_size:
                for i in range(total_categories):
                    if indices[i] < len(category_data[i]):
                        selected_data.append(category_data[i][indices[i]])
                        indices[i] += 1
                    if len(selected_data) == target_size:
                        break
                        
                if min(indices) >= len(category_data[0]):
                    break
              
            for i in range(len(indices)):
                indices[i] += sequence_length
                indices[i] -= 1
            return selected_data, indices

        # Initialize indices for each category
        start_indices = [0] * total_categories

        # Fill the train, val, and test sets
        train_data, start_indices = round_robin_select(train_size, start_indices)
        val_data, start_indices = round_robin_select(val_size, start_indices)
        remaining_size = total_frames - len(train_data) - len(val_data)
        test_data, start_indices = round_robin_select(remaining_size, start_indices)
 
        # Ensure no overlaps
        train_ids = {item['id'] for sequence in train_data for item in sequence}
        val_ids = {item['id'] for sequence in val_data for item in sequence}
        test_ids = {item['id'] for sequence in test_data for item in sequence}

        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)

        if train_val_overlap:
            raise ValueError(f"Overlap between train and val: {train_val_overlap}")
        if train_test_overlap:
            raise ValueError(f"Overlap between train and test: {train_test_overlap}")
        if val_test_overlap:
            raise ValueError(f"Overlap between val and test: {val_test_overlap}")
        else:
            print('No overlaps found in data')
    else:
        random.shuffle(data)
        train_data, val_test_data = train_test_split(data, train_size=0.7, random_state=42)
        val_data, test_data = train_test_split(val_test_data, train_size=0.5, random_state=42)
    
    return train_data, val_data, test_data

def get_datasets(json_files: List[str], base_path: str, sequence_length: Optional[int] = None) -> Tuple[TennisDataset, TennisDataset, TennisDataset]:
    all_data = load_annotations(json_files, base_path)
    transform_train = get_train_transform()
    transform_val = get_val_transform()
    
    if sequence_length:
        # Sequentialize first
        sequences = sequentialize_data(all_data, sequence_length)
        train_data, val_data, test_data = split_data(sequences, sequence_length)
    else:
        # Split first
        train_data, val_data, test_data = split_data(all_data)
    train_dataset = TennisDataset(train_data, transform=transform_train, sequence_length=sequence_length)
    val_dataset = TennisDataset(val_data,transform=transform_val, sequence_length=sequence_length)
    test_dataset = TennisDataset(test_data,transform=transform_val, sequence_length=sequence_length)
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    json_files = ['backhand.json', 'forehand.json', 'serve.json', 'ready_position.json']
    base_path = "og_dataset"
    # Non-sequential data
    train_dataset, val_dataset, test_dataset = get_datasets(json_files, base_path)
    print(f"Non-sequential - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_dataset_seq, val_dataset_seq, test_dataset_seq = get_datasets(json_files, base_path, sequence_length=5)
    print(f"Sequential - Train: {len(train_dataset_seq)}, Val: {len(val_dataset_seq)}, Test: {len(test_dataset_seq)}")
    
    # Print a sample of the data for each dataset
    def print_sample(dataset, name):
        sample = dataset[0]
        if dataset.sequence_length:
            frames, bboxes, keypoints, label = sample
            print(f"{name} - Sequence Sample:")
            print(f"Frames shape: {frames.shape}")
            print(f"BBoxes shape: {bboxes.shape}")
            print(f"Keypoints shape: {keypoints.shape}")
            print(f"Label: {label}")
        else:
            image, bboxes, keypoints, label = sample
            print(f"{name} - Sample:")
            print(f"Image shape: {image.shape}")
            print(f"BBoxes: {bboxes}")
            print(f"Keypoints: {keypoints}")
            print(f"Label: {label}")
        print('\n')

    print_sample(train_dataset, "Train Dataset")
    print_sample(val_dataset, "Validation Dataset")
    print_sample(test_dataset, "Test Dataset")
    print('\n')
    print_sample(train_dataset_seq, "Train Dataset Sequential")
    print_sample(val_dataset_seq, "Validation Dataset Sequential")
    print_sample(test_dataset_seq, "Test Dataset Sequential")

