import torch
import torch.nn as nn
import torchvision.models as models
import sys
import torch.nn.functional as F

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

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class TennisNet(nn.Module):
    def __init__(self, num_keypoints=18, num_classes=4, backbone_name='efficientnet_b3'):
        super(TennisNet, self).__init__()
        
        # Backbone
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
        
        # Spatial Attention
        self.spatial_attention = SpatialAttention(self.backbone_channels)
        
        # Temporal modeling
        self.conv3d = nn.Conv3d(self.backbone_channels, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.lstm = nn.LSTM(256, hidden_size=256, num_layers=2, batch_first=True)
        
        # Temporal Attention
        self.temporal_attention = TemporalAttention(256)
        
        # Prediction heads
        self.keypoint_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_keypoints * 3)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        try:
            # print(f"Input shape: {x.shape}")  # Debugging: Print input shape
            batch_size, seq_len, c, h, w = x.size()
            
            # Reshape for the backbone (merge batch and sequence dimensions)
            x = x.view(batch_size * seq_len, c, h, w)
            # print(f"Reshaped input for backbone: {x.shape}")  # Debugging: Print reshaped input shape
            
            # Extract spatial features using the backbone
            features = self.backbone(x)  # (batch_size * seq_len, backbone_channels, H', W')
            # print(f"Features shape after backbone: {features.shape}")  # Debugging: Print features shape after backbone
            
            # Apply spatial attention
            features = self.spatial_attention(features)  # (batch_size * seq_len, backbone_channels, H', W')
            # print(f"Features shape after spatial attention: {features.shape}")  # Debugging: Print features shape after spatial attention
            
            # Global average pooling on spatial dimensions
            features = features.mean(dim=[2, 3])  # (batch_size * seq_len, backbone_channels)
            # print(f"Features shape after global pooling: {features.shape}")  # Debugging: Print features shape after global pooling
            
            # Reshape back to sequence form
            features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, backbone_channels)
            # print(f"Features shape after reshaping to sequence: {features.shape}")  # Debugging: Print reshaped features
            
            # Temporal modeling using 3D convolution
            features = features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, backbone_channels, seq_len, 1, 1)
            features = self.conv3d(features)  # (batch_size, 256, seq_len, 1, 1)
            features = features.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (batch_size, seq_len, 256)
            # print(f"Features shape after 3D convolution: {features.shape}")  # Debugging: Print features shape after 3D convolution
            
            # Temporal modeling using LSTM
            lstm_out, _ = self.lstm(features)  # (batch_size, seq_len, 256)
            # print(f"LSTM output shape: {lstm_out.shape}")  # Debugging: Print LSTM output shape
            
            # Temporal attention
            attn_out = self.temporal_attention(lstm_out)  # (batch_size, seq_len, 256)
            # print(f"Attention output shape: {attn_out.shape}")  # Debugging: Print attention output shape
            
            # Use the last output in the sequence for predictions
            final_features = attn_out[:, -1, :]  # (batch_size, 256)
            # print(f"Final features shape: {final_features.shape}")  # Debugging: Print final features shape
            
            # Predictions
            keypoints = self.keypoint_head(final_features)  # (batch_size, num_keypoints * 3)
            bboxes = self.bbox_head(final_features)  # (batch_size, 4)
            classification_logits = self.classification_head(final_features)  # (batch_size, num_classes)
            
            return keypoints, bboxes, classification_logits
        except RuntimeError as e:
            # print(f"RuntimeError: {e}")
            # print(f"Shape mismatch at some layer. Input shape: {x.shape}")
            raise

    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = Config.get_device()
    model = TennisNet().to(device)
    # print number of trainable parameters in the model
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    input_tensor = torch.randn(16, 16, 3, 320, 320).to(device)  # Batch size 32, sequence length 16
    keypoints, bboxes, classification_logits = model(input_tensor)

    print(f"Keypoints: {keypoints.shape}")  # Expected: torch.Size([32, 54])
    print(f"BBoxes: {bboxes.shape}")  # Expected: torch.Size([32, 4])
    print(f"Classification Logits: {classification_logits.shape}")  # Expected: torch.Size([32, 4])
