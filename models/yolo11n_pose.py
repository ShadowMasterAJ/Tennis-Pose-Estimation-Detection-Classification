import itertools
import os
from ultralytics import YOLO

# Define the hyperparameter grid
hyperparameter_grid = {
    'batch': [32, 64],
    'lr0': [0.001, 0.0005],
    'momentum': [0.9],
    'scale': [0.5, 1.0],
    'translate': [0.1],
    'shear': [2.0, 5.0],
}

# Generate all possible combinations of hyperparameters
combinations = list(itertools.product(*hyperparameter_grid.values()))

# Initialize the YOLO model
model = YOLO(model='YOLO11m' 'downloaded_models\yolo11m-pose.pt')

# Loop through each combination and run the training process
for i, combination in enumerate(combinations):
    hyperparameters = dict(zip(hyperparameter_grid.keys(), combination))
    print(f"Running training for combination {i + 1}/{len(combinations)}: {hyperparameters}")
    
    project_name = f'tennis_pose_estimation_m_{i}'
    results = model.train(
        data='dataset_yolo/dataset.yaml',
        epochs=300,  # Fixed at 300
        batch=hyperparameters['batch'],
        imgsz=640,
        device=0,
        optimizer='AdamW',
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

    # Save hyperparameters to config.txt
    config_path = os.path.join('tennis_pose_estimation', project_name, 'config.txt')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write("epochs: 300\n")
        f.write("imgsz: 640\n")
        f.write("device: 0\n")
        f.write("optimizer: AdamW\n")
        f.write("patience: 50\n")
        f.write("save: True\n")
        f.write("save_period: 50\n")
        f.write("cache: disk\n")
        f.write("project: tennis_pose_estimation\n")
        f.write("pretrained: True\n")
        f.write("verbose: True\n")
        f.write("seed: 42\n")
        f.write("close_mosaic: 15\n")
        f.write("freeze: 11\n")
        f.write("lrf: 0.01\n")
        f.write("cos_lr: True\n")
        f.write("warmup_epochs: 3\n")
        f.write("warmup_momentum: 0.8\n")
        f.write("warmup_bias_lr: 0.1\n")
        f.write("box: 7\n")
        f.write("cls: 1\n")
        f.write("dfl: 2\n")
        f.write("pose: 15.0\n")
        f.write("kobj: 20.0\n")
        f.write("plots: True\n")
        f.write("augment: True\n")
        f.write("val: True\n")
        f.write("mixup: 0.1\n")
        f.write("copy_paste: 0.1\n")
        f.write("degrees: 10.0\n")
        f.write("perspective: 0.0005\n")
        f.write("mosaic: 1.0\n")

    print(f"Finished training for combination {i + 1}/{len(combinations)}")

print("Hyperparameter grid search completed.")