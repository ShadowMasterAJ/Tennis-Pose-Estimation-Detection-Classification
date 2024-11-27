import torch
import torch.nn as nn
from torchvision import models
from training.config import Config
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels//2, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels//2, 1)
        self.conv_out = nn.Conv2d(in_channels//2 * 3, in_channels, 1)

    def forward(self, x):
        # Assuming x is the largest feature map
        x1 = self.conv1(x)
        x2 = self.conv2(F.avg_pool2d(x, 2))
        x3 = self.conv3(F.avg_pool2d(x, 4))

        # Upsample x2 and x3 to match x1's size
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate along the channel dimension
        out = torch.cat([x1, x2, x3], dim=1)
        
        # Final convolution to merge features
        out = self.conv_out(out)

        return out
class EnhancedTennisConv(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b3'):
        super(EnhancedTennisConv, self).__init__()
        backbone_config = Config.get_backbone_layers(backbone_name)
        freeze_backbone = backbone_config['freeze_layers']

        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")
        
        self.backbone = self._get_backbone(backbone_name)
        self.backbone_channels =  backbone_config['output_channels']
        
        if freeze_backbone:
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.se_block = SEBlock(self.backbone_channels)
        self.multi_scale_fusion = MultiScaleFusion(self.backbone_channels)
        self.num_keypoints = num_keypoints
        self.num_keypoint_outputs = num_keypoints * 3
        self.keypoint_head = nn.Sequential(
            # Input: (batch_size, backbone_channels, H, W)
            nn.Conv2d(self.backbone_channels, 512, kernel_size=3, padding=1),  # Output: (batch_size, 512, H, W)
            nn.BatchNorm2d(512),  # Output: (batch_size, 512, H, W)
            nn.ReLU(),  # Output: (batch_size, 512, H, W)
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Output: (batch_size, 256, H, W)
            nn.BatchNorm2d(256), # Output: (batch_size, 256, H, W)
            nn.ReLU(), # Output: (batch_size, 256, H, W)
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # Output: (batch_size, 128, H, W)
            nn.BatchNorm2d(128), # Output: (batch_size, 128, H, W)
            nn.ReLU(), # Output: (batch_size, 128, H, W)
            nn.AdaptiveAvgPool2d(1), # Output: (batch_size, 128, 1, 1)
            nn.Flatten(), # Output: (batch_size, 128)
            nn.Linear(128, self.num_keypoint_outputs), # Output: (batch_size, num_keypoints * 3)
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(self.backbone_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 4),  # 4 for [x, y, width, height]
            nn.Sigmoid()  # Normalize bbox coordinates
        )

        self.class_head = nn.Sequential(
            nn.Conv2d(self.backbone_channels, 512, kernel_size=3, padding=1),
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

    def forward(self, x):
        features = self.backbone(x)
        
        # Apply SE block
        features = self.se_block(features)
        
        # Multi-scale feature fusion
        fused_features = self.multi_scale_fusion(features)
        
        # Keypoint prediction
        keypoints = self.keypoint_head(fused_features)
        
        # Bounding box prediction
        bboxes = self.bbox_head(fused_features)
        
        # Class prediction
        class_output = self.class_head(fused_features)
        
        return keypoints, bboxes, class_output

    def _get_backbone(self, name):
        if name == 'efficientnet_b3':
            return models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1).features
        elif name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(backbone.children())[:-2])
        elif name == 'efficientnet_b7':
            return models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1).features
        elif name == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {name}")

class TennisBRNN(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b3', hidden_size=256, num_layers=2):
        super(TennisBRNN, self).__init__()
        
        print("Initialising the model!")
        self.num_keypoints = num_keypoints
        self.num_keypoint_outputs = num_keypoints * 3
        
        backbone_config = Config.get_backbone_layers(backbone_name)
        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")
        
        self.backbone = self._get_backbone(backbone_name)
        self.backbone_channels = backbone_config['output_channels']
        
        # Optionally freeze the backbone
        if backbone_config['freeze_layers']:
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.se_block = SEBlock(self.backbone_channels)
        self.multi_scale_fusion = MultiScaleFusion(self.backbone_channels)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(input_size=self.backbone_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Keypoint Prediction Head
        self.keypoint_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_keypoint_outputs)
        )
        
        # Bounding Box Head
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        
        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def _get_backbone(self, name):
        if name == 'efficientnet_b3':
            backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.backbone_channels = 1536
            return nn.Sequential(*list(backbone.children())[:-2])
        elif name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone_channels = 2048
            return nn.Sequential(*list(backbone.children())[:-2])
        elif name == 'efficientnet_b7':
            backbone = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            self.backbone_channels = 2560
            return nn.Sequential(*list(backbone.children())[:-2])
        elif name == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.backbone_channels = 2048
            return nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x):
        batch_size, seq_len, h, w = x.size() # (batch_size, seq_len, H, W)
        x = x.view(batch_size * seq_len, h, w) # (batch_size * seq_len, H, W)
        
        features = self.backbone(x) # (batch_size * seq_len, C, H, W)
        features = self.se_block(features) # Apply SE block
        fused_features = self.multi_scale_fusion(features) # Multi-scale feature fusion
        
        fused_features = fused_features.view(batch_size, seq_len, -1) # (batch_size, seq_len, C)
        
        lstm_out, _ = self.bilstm(fused_features) # (batch_size, seq_len, hidden_size * 2)
        
        keypoints = self.keypoint_head(lstm_out[:, -1, :]) # (batch_size, num_keypoints * 3)
        bboxes = self.bbox_head(lstm_out[:, -1, :]) # (batch_size, 4)
        classification_logits = self.classification_head(lstm_out[:, -1, :]) # (batch_size, num_classes)
        
        return keypoints, bboxes, classification_logits

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
class TennisConvResidual(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b7'):
        super(TennisConvResidual, self).__init__()
        
        print("Initialising the model!")
        self.num_keypoints = num_keypoints
        self.num_keypoint_outputs = num_keypoints * 3
        
        backbone_config = Config.get_backbone_layers(backbone_name)
        freeze_backbone = backbone_config['freeze_layers']

        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")
        
        self.backbone = self._get_backbone(backbone_name)
        
        if freeze_backbone:
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze the last few layers
            for param in list(self.backbone.parameters())[-10:]:
                param.requires_grad = True
        
        self.keypoint_head = nn.Sequential(
            ResidualBlock(backbone_config['output_channels'], 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.num_keypoint_outputs),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        self.bbox_head = nn.Sequential(
            ResidualBlock(backbone_config['output_channels'], 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        self.classification_head = nn.Sequential(
            ResidualBlock(backbone_config['output_channels'], 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
            nn.Dropout(Config.DROPOUT_RATE)
        )
    
    def _get_backbone(self, name):
        if name == 'efficientnet_b3':
            return models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1).features
        elif name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(backbone.children())[:-2])
        elif name == 'efficientnet_b7':
            return models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1).features
        elif name == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x):
        features = self.backbone(x)
        
        keypoints = self.keypoint_head(features)
        bboxes = self.bbox_head(features)
        classification_logits = self.classification_head(features)
        
        return keypoints, bboxes, classification_logits
    
class TennisConvV1(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b3'):
        super(TennisConvV1, self).__init__()
        
        print("Initialising the model!")
        self.num_keypoints = num_keypoints
        self.num_keypoint_outputs = num_keypoints * 3
        
        backbone_config = Config.get_backbone_layers(backbone_name)
        freeze_backbone = backbone_config['freeze_layers']

        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")
        
        self.backbone = self._get_backbone(backbone_name)
        
        if freeze_backbone:
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze the last few layers
            for param in list(self.backbone.parameters())[-10:]:
                param.requires_grad = True
        
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.num_keypoint_outputs),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4),
            nn.Dropout(Config.DROPOUT_RATE)
        )
        
        self.classification_head = nn.Sequential(
            nn.Conv2d(backbone_config['output_channels'], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
            nn.Dropout(Config.DROPOUT_RATE)
        )

    def _get_backbone(self, name):
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
    

class TennisConv(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b7',freeze_backbone=True):
        super(TennisConv, self).__init__()
        
        print("Initialising the model!")
        # Define number of keypoints and their format (x, y, v)
        self.num_keypoints = num_keypoints
        self.num_keypoint_outputs = num_keypoints * 3  # Each keypoint has x, y, and visibility
        
        # Backbone configuration
        backbone_config = Config.get_backbone_layers(backbone_name)
        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")
        
        self.backbone = self._get_backbone(backbone_name)
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
    def _get_backbone(self, name):
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