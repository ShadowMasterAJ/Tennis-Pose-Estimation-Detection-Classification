import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Add the project to the path for importing custom modules
sys.path.append('C:\\Users\\arnav\\Documents\\University\\CS 5100 Foundations of Artificial Intelligence\\Final Project\\Final Project')

from training.config import Config

class TennisPoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b0'):
        super(TennisPoseEstimationModel, self).__init__()
        self.num_keypoints = num_keypoints
        backbone_config = Config.get_backbone_layers(backbone_name)
        if backbone_config is None:
            raise ValueError(f"Unknown backbone model: {backbone_name}")

        self.backbone = self._get_backbone(backbone_name)
        self.backbone_channels = backbone_config['output_channels']

        # Optionally freeze the backbone
        if backbone_config.get('freeze_layers', False):
            print(f"Freezing the backbone layers of {backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.backbone_channels,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            bidirectional=True
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
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # x, y, width, height
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def _get_backbone(self, name):
        if name == 'efficientnet_b0':
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        # Add more backbone options if needed

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        features = self.backbone(x)
        features = features.mean(dim=[2, 3]).view(batch_size, seq_len, -1)

        lstm_out, _ = self.bilstm(features)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for prediction
        final_features = attn_output[:, -1, :]

        keypoints = self.keypoint_head(final_features)
        bboxes = self.bbox_head(final_features)
        classification_logits = self.classification_head(final_features)
        classification = torch.argmax(F.softmax(classification_logits, dim=1), dim=1)

        return keypoints, bboxes, classification
    
    
if __name__ == "__main__":
    device = Config.get_device()
    model = TennisPoseEstimationModel().to(device)

    input_tensor = torch.randn(32, 16, 3, 320, 320).to(device)  # Batch size 32, sequence length 16
    keypoints, bboxes, classification_logits = model(input_tensor)

    print(f"Keypoints: {keypoints.shape}")  # Expected: torch.Size([32, 54])
    print(f"BBoxes: {bboxes.shape}")  # Expected: torch.Size([32, 4])
    print(f"Classification Logits: {classification_logits.shape}")  # Expected: torch.Size([32, 4])
