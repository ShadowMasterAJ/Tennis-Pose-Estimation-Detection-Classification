import torch.nn as nn
from torchvision import models
from training.config import Config

class TennisConv(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b3'):
        super(TennisConv, self).__init__()
        
        print("Initialising the model!")
        self.num_keypoints = num_keypoints
        self.num_keypoint_outputs = num_keypoints * 3
        
        backbone_config = Config.get_backbone_layers(backbone_name)
        freeze_backbone = backbone_config['freeze_layers']

        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")
        
        self.backbone = self._get_backbone(backbone_name, backbone_config['pretrained'])
        
        if freeze_backbone:
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, self.num_keypoint_outputs),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 4),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        self.classification_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
            nn.Dropout(Config.DROPOUT_RATE)
        )

    def _get_backbone(self, name, pretrained):
        if name == 'efficientnet_b3':
            return models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1).features
        elif name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(backbone.children())[:-2])
        elif name == 'efficientnet_b7':
            return models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1).features
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x):
        features = self.backbone(x)
        
        keypoints = self.keypoint_head(features)
        bboxes = self.bbox_head(features)
        classification_logits = self.classification_head(features)
        
        return keypoints, bboxes, classification_logits