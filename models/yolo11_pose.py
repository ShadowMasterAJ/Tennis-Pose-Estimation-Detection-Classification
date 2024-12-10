import itertools
import os
from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    hyperparameter_grid = {
        'lr0': [0.001, 0.0005],  # Learning rates
        'optimizer': ['Adam', 'SGD'],  # Optimizers
        'imgsz': [640, 320]  # Image sizes
    }
    # Generate all possible combinations of hyperparameters
    combinations = list(itertools.product(*hyperparameter_grid.values()))

    # Initialize the YOLO model
    model = YOLO(model='downloaded_models/yolo11m-pose.pt')

    # Loop through each combination and run the training process
    for i, combination in enumerate(combinations):
        hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))
        print(f"Running training for combination {i + 1}/{len(combinations)}: {hyperparameters}")
        
        project_name = f'training_run_{i}'
        project_name = f"lr{str(hyperparameters['lr0']).replace('.', '')}_opt{hyperparameters['optimizer']}_imgsz{hyperparameters['imgsz']}"
        
        results = model.train(
            data='dataset_yolo/dataset.yaml',
            epochs=300,  # Fixed at 300
            batch=32,
            imgsz=hyperparameters['imgsz'],
            device=0,
            optimizer=hyperparameters['optimizer'],
            lr0=hyperparameters['lr0'],
            patience=20,
            name=project_name,
            save=True,
            save_period=50,
            cache='disk',
            project='tennis_pose_estimation',
            pretrained=True,
            verbose=True,
            seed=42,
            close_mosaic=15,
            freeze=11,
            lrf=0.01,
            cos_lr=True,
            momentum=0.9,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7,
            cls=1,
            dfl=2,
            pose=15.0,
            kobj=20.0,
            plots=True,
            augment=True,
            val=True,
            mixup=0.1,
            copy_paste=0.1,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=3,
            perspective=0.0005,
            mosaic=1.0
        )

        print(f"Finished training for combination {i + 1}/{len(combinations)}")

    print("Hyperparameter grid search completed.")

if __name__ == '__main__':
    mp.freeze_support()
    main()