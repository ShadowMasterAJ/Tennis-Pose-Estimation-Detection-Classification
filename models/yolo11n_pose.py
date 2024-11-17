from ultralytics import YOLO
import os, shutil, multiprocessing
from torchview import draw_graph
import torch
import sys
import io
def main():
    model = YOLO('tennis_pose_estimation/tennis_pose_estimation_nano2/weights/best.pt')
    
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

    # results = model.train(
    #     data='dataset_yolo/dataset.yaml',
    #     epochs=300,
    #     batch=32,
    #     imgsz=640,
    #     device=0,
    #     optimizer='AdamW',
    #     lr0=0.001,
    #     patience=50,
    #     name='tennis_pose_estimation_nano',
    #     save=True,
    #     save_period=100,
    #     cache='disk',
    #     project='tennis_pose_estimation',
    #     pretrained=True,
    #     verbose=True,
    #     seed=42,
    #     close_mosaic=15,
    #     amp=True,
    #     lrf=0.01,
    #     cos_lr=True,
    #     momentum=0.937,
    #     warmup_epochs=3,
    #     warmup_momentum=0.8,
    #     warmup_bias_lr=0.1,
    #     box=12,
    #     cls=2,
    #     dfl=2,
    #     pose=20.0,
    #     kobj=20.0,
    #     plots=True,
    #     augment=True,
    #     mixup=0.1,
    #     copy_paste=0.1,
    #     degrees=10.0,
    #     translate=0.1,
    #     scale=0.5,
    #     shear=2.0,
    #     perspective=0.0005,
    #     flipud=0.5,
    #     fliplr=0.5,
    #     mosaic=1.0
    # )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()