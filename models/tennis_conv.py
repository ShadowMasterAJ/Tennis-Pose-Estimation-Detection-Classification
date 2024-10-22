import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from training.config import Config

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class TennisConv(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b3',freeze_backbone=True):
        super(TennisConv, self).__init__()
        
        print("Initialising the model!")
        # Define number of keypoints and their format (x, y, v)
        self.num_keypoints = num_keypoints
        self.num_keypoint_outputs = num_keypoints * 3  # Each keypoint has x, y, and visibility
        
        # Backbone configuration
        backbone_config = Config.get_backbone_layers(backbone_name)
        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")
        
        self.backbone = self._get_backbone(backbone_name, backbone_config['pretrained'])
        if freeze_backbone:
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Keypoint Prediction Head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(256, self.num_keypoint_outputs)  # Predict 54 values (18 keypoints * 3)
        )
        
        # Bounding Box Head (predict 4 coordinates)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 4)  # Predict (x1, y1, x2, y2)
        )
        
        # Classification Head (for shot type)
        self.classification_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)  # Predict class logits
        )

    def _get_backbone(self, name, pretrained):
        """Select backbone based on the model name."""
        if name == 'efficientnet_b3':
            return models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1).features
        elif name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(backbone.children())[:-2])  # Remove the classifier head
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x):
        features = self.backbone(x)
        
        # Keypoint prediction (18 keypoints, each with x, y, v)
        keypoints = self.keypoint_head(features)
        
        # Bounding box prediction (4 values)
        bboxes = self.bbox_head(features)
        
        # Classification prediction (4 shot types)
        classification_logits = self.classification_head(features)
        
        return keypoints, bboxes, classification_logits
