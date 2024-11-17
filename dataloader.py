import json
import os
import random
from sklearn.model_selection import train_test_split
from utils.data_utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.visualiser import visualise_dataloader_sample,visualise_dataset

def prepare_data(json_files, base_path):
    all_data = []
    
    for json_file in json_files:
        all_data.extend(load_annotations(json_file, base_path))
    
    return split_data(all_data)

def load_annotations(json_file, base_path):
    """ Load annotations from a single JSON file. """
    # shot_type = os.path.splitext(os.path.basename(json_file))[0]
    json_file_path = os.path.join(base_path, 'annotations', json_file)

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file {json_file_path} not found.")
        return []

    images_data = []
    shot_type = data['categories'][0]['name'].lower()
    for img_info in data['images']:
        img_path = os.path.join(base_path, img_info['path'].lstrip('../'))
        annotation = next((ann for ann in data['annotations'] if ann['image_id'] == img_info['id']), None)

        if annotation:
            bbox = annotation['bbox']
            keypoints = annotation['keypoints']

            images_data.append({
                'image_path': img_path,
                'bbox': bbox,
                'keypoints': keypoints,
                'label': shot_type  # Changed from shot_type to label for consistency
            })

    return images_data

def split_data(all_data):
    random.shuffle(all_data)
    
    train_data, val_data, test_data = [], [], []
    for shot_type in set(item['label'] for item in all_data):
        shot_data = [item for item in all_data if item['label'] == shot_type]
        train, val_test = train_test_split(shot_data, train_size=0.7, random_state=42)
        val, test = train_test_split(val_test, train_size=0.67, test_size=0.33, random_state=42)
        train_data.extend(train)
        val_data.extend(val)
        test_data.extend(test)
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def get_datasets(json_files, base_path):
    train_data, val_data, test_data = prepare_data(json_files, base_path)
    transform = get_transform()

    train_dataset = TennisDataset(train_data, transform=transform)
    val_dataset = TennisDataset(val_data, transform=transform)
    test_dataset = TennisDataset(test_data, transform=transform)

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(json_files, base_path, batch_size=32):
    train_dataset, val_dataset, test_dataset = get_datasets(json_files, base_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    json_files = os.listdir("og_dataset\\annotations")
    base_path = "og_dataset"
    train_loader, val_loader, test_loader = get_dataloaders(json_files, base_path)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, bboxes, keypoints, labels = next(iter(train_loader))
    print(f"Sample batch - Images shape: {images.shape}")
    print(f"Sample batch - Bounding boxes shape: {bboxes.shape}")
    print(f"Sample batch - Keypoints shape: {keypoints.shape}")
    print(f"Sample batch - Labels shape: {labels.shape}")

    # # Plot sample from the dataloader
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sample_idx = 13
    # visualise_dataloader_sample(ax, images[sample_idx], bboxes[sample_idx], keypoints[sample_idx], labels[sample_idx], "Sample from DataLoader")
    # plt.show()

    
    # Visualize dataset samples
    for json_file in json_files:
        fig = visualise_dataset(f"{base_path}/annotations/{json_file}", base_path, num_images=2)
        plt.show()
    
    