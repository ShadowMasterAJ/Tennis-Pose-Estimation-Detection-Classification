import torch

class Config:
    # Training Parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    
    LOG_DIR = 'logs'

    # Model Architecture Parameters
    NUM_KEYPOINTS = 18  # Number of keypoints to predict
    NUM_CLASSES = 4  # Number of shot classes
    BACKBONE = 'efficientnet_b3'  # Backbone model

    # Model Layer Configurations
    BACKBONE_LAYERS = {
        'efficientnet_b3': {
            'output_channels': 1536,  # Adjust according to the selected backbone
            'pretrained': True,
        },
        'resnet50': {
            'output_channels': 2048,
            'pretrained': True,
        },
        # Add other backbones as needed
    }

    # Loss Function Weights
    LOSS_WEIGHTS = {
        'keypoints': 1.0,
        'bbox': 1.0,
        'classification': 1.0
    }

    @staticmethod
    def get_backbone_layers(backbone_name):
        """Retrieve layer details for the specified backbone."""
        return Config.BACKBONE_LAYERS.get(backbone_name, None)

    @staticmethod
    def get_device():
        """Get the device for training."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
