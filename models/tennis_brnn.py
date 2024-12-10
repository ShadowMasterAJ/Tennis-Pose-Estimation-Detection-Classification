import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Add the project to the path for importing custom modules
sys.path.append('C:\\Users\\arnav\\Documents\\University\\CS 5100 Foundations of Artificial Intelligence\\Final Project\\Final Project')

from training.config import Config

def get_backbone(name):
    if name == 'efficientnet_b0':
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
    elif name == 'efficientnet_b3':
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

class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels

    def forward(self, x):
        batch_size, c, h, w = x.size()
        pooling_layers = []
        for level in self.num_levels:
            pooling = nn.AdaptiveMaxPool2d(output_size=(level, level))
            pooling_layers.append(pooling(x).view(batch_size, -1))
        return torch.cat(pooling_layers, dim=1)

class TennisPoseSPP(nn.Module):
    def __init__(self,backbone_name='efficientnet_b3'):
        super(TennisPoseSPP, self).__init__()
        self.num_keypoints = Config.NUM_KEYPOINTS
        self.num_classes = Config.NUM_CLASSES
        backbone_config = Config.get_backbone_layers(backbone_name)
        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")

        self.backbone = get_backbone(backbone_name)
        self.backbone_channels = backbone_config['output_channels']

        # Optionally freeze the backbone
        if backbone_config.get('freeze_layers', False):
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Spatial Pyramid Pooling
        self.spp = SPPLayer([1, 2, 4])

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.backbone_channels * 21,  # Adjusted for SPP output
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        # Keypoint Prediction Head
        self.keypoint_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, self.num_keypoints * 3)
        )

        # Bounding Box Head
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, 4)  # x, y, width, height
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        features = self.backbone(x)
        spp_features = self.spp(features)
        spp_features = spp_features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.bilstm(spp_features)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for prediction
        final_features = attn_output[:, -1, :]

        keypoints = self.keypoint_head(final_features)
        bboxes = self.bbox_head(final_features)
        classification_logits = self.classification_head(final_features)

        return keypoints, bboxes, classification_logits

class TennisPoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b0'):
        super(TennisPoseEstimationModel, self).__init__()
        self.num_keypoints = num_keypoints
        backbone_config = Config.get_backbone_layers(backbone_name)
        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")

        self.backbone = get_backbone(backbone_name)
        self.backbone_channels = backbone_config['output_channels']

        # Optionally freeze the backbone
        if backbone_config.get('freeze_layers', False):
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.backbone_channels,
            hidden_size=32,
            num_layers=4,
            batch_first=True,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)

        # Keypoint Prediction Head
        self.keypoint_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, num_keypoints * 3)
        )

        # Bounding Box Head
        self.bbox_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # x, y, width, height
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        features = self.backbone(x)
        features = features.mean(dim=[2, 3]).view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for prediction
        final_features = attn_output[:, -1, :]

        keypoints = self.keypoint_head(final_features)
        bboxes = self.bbox_head(final_features)
        classification_logits = self.classification_head(final_features)

        return keypoints, bboxes, classification_logits


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = Config.get_device()
    model = TennisPoseSPP().to(device)
    # print number of trainable parameters in the model
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    input_tensor = torch.randn(64, 5, 3, 320, 320).to(device)  # Batch size 32, sequence length 5
    keypoints, bboxes, classification_logits = model(input_tensor)

    print(f"Keypoints: {keypoints.shape}")  # Expected: torch.Size([32, 54])
    print(f"BBoxes: {bboxes.shape}")  # Expected: torch.Size([32, 4])
    print(f"Classification Logits: {classification_logits.shape}")  # Expected: torch.Size([32, 4])
    
    print('Sample output:')
    print(keypoints[0])
    print(bboxes[0])
    probabilities = torch.softmax(classification_logits, dim=1)
    print(probabilities[0])