import itertools
from ultralytics import YOLO
import os, shutil, multiprocessing
from torchview import draw_graph
import torch
import sys
import io
def main():
    
    # print(model.info(detailed=True))
    
    # # Write model architecture to a text file
    # with open('model_nano_pose.txt', 'w') as f:
    #     f.write("Model Architecture: YOLOv8n-pose\n\n")
    #     f.write("Sequential(\n")
    #     for i, (name, module) in enumerate(model.model.named_children()):
    #         f.write(f"  ({i}): {name} (\n")
    #         if hasattr(module, 'named_children'):
    #             for j, (sub_name, sub_module) in enumerate(module.named_children()):
    #                 f.write(f"    ({j}): {sub_name}: {sub_module}\n")
    #         else:
    #             f.write(f"    {module}\n")
    #         f.write("  )\n")
    #     f.write(")\n")
    #     f.write(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}\n")
    #     f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    model = YOLO(model='YOLO11m' 'downloaded_models\yolo11m-pose.pt')

    # Define the hyperparameter grid
    hyperparameter_grid = {
        'epochs': [100, 200, 300],
        'batch': [32, 64],
        'lr0': [0.001, 0.0005],
        'momentum': [0.9, 0.937],
        'scale': [0.5, 1.0],
        'translate': [0.1, 0.2],
        'shear': [2.0, 5.0],
    }

    # Generate all possible combinations of hyperparameters
    combinations = list(itertools.product(*hyperparameter_grid.values()))

    # Loop through each combination and run the training process
    for i, combination in enumerate(combinations):
        hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))
        print(f"Running training for combination {i + 1}/{len(combinations)}: {hyperparameters}")
        
        results = model.train(
            data='dataset_yolo/dataset.yaml',
            epochs=hyperparameters['epochs'],
            batch=hyperparameters['batch'],
            imgsz=640,
            device=0,
            optimizer='AdamW',
            lr0=hyperparameters['lr0'],
            patience=50,
            name=f'tennis_pose_estimation_m_{i}',
            save=True,
            save_period=50,
            cache='disk',
            project='tennis_pose_estimation',
            pretrained=True,
            verbose=True,
            seed=42,
            close_mosaic=15,
            freeze = 11,
            lrf=0.01,
            cos_lr=True,
            momentum=hyperparameters['momentum'],
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
            translate=hyperparameters['translate'],
            scale=hyperparameters['scale'],
            shear=hyperparameters['shear'],
            perspective=0.0005,
            mosaic=1.0
        )

        # Save or log the results as needed
        print(f"Finished training for combination {i + 1}/{len(combinations)}")

    print("Hyperparameter grid search completed.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()