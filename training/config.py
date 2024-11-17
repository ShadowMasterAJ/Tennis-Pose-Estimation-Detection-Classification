import torch

class Config:
    # General Parameters
    SEED = 42  # For reproducibility

    # Training Parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10  # Early stopping patience
    GRAD_CLIP = 1.0  # Gradient clipping to prevent exploding gradients
    RESUME_TRAINING = False  # If True, resume from checkpoint

    # Optimizer Parameters
    OPTIMIZER = 'adamw'  # Options: 'adam', 'sgd', 'adamw'
    MOMENTUM = 0.9  # Momentum factor (for SGD optimizer)
    WEIGHT_DECAY = 1e-5  # L2 regularization (weight decay)
    
    # Learning Rate Scheduler Parameters
    LR_SCHEDULER = 'CosineAnnealingLR'  # Options: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR'
    LR_SCHEDULER_PATIENCE = 5  # ReduceLROnPlateau patience before reducing LR
    LR_SCHEDULER_FACTOR = 0.1  # ReduceLROnPlateau LR reduction factor
    LR_STEP_SIZE = 10  # For StepLR, steps before reducing LR
    LR_GAMMA = 0.1  # LR reduction factor for StepLR
    
    LOG_DIR = 'logs'
    
    # Regularization Parameters
    DROPOUT_RATE = 0.5  # Dropout rate for model layers
    BATCH_NORM = True  # Use batch normalization or not

    # Model Architecture Parameters
    NUM_KEYPOINTS = 18  # Number of keypoints to predict
    NUM_CLASSES = 4  # Number of shot classes
    BACKBONE = 'efficientnet_b3'  # Backbone model

    # Model Layer Configurations
    BACKBONE_LAYERS = {
        'efficientnet_b3': {
            'output_channels': 1536,  # Adjust according to the selected backbone
            'pretrained': True,
            'freeze_layers': True,  # Whether to freeze the backbone layers
        },
        'resnet50': {
            'output_channels': 2048,
            'pretrained': True,
            'freeze_layers': True,  # If False, fine-tune the ResNet backbone
        },
        'efficientnet_b7': {
            'output_channels': 2560,
            'pretrained': True,
            'freeze_layers': True,
        },
        # Add other backbones as needed
    }

    # Loss Function Weights
    LOSS_WEIGHTS = {
        'keypoints': 1.0,
        'bbox': 1.0,
        'classification': 1.0
    }

    # Checkpointing Parameters
    CHECKPOINT_DIR = 'checkpoints'
    SAVE_BEST_ONLY = True  # If True, only save the best-performing model
    SAVE_EVERY_EPOCH = True  # If True, save the model after every epoch
    
    @staticmethod
    def get_backbone_layers(backbone_name):
        """Retrieve layer details for the specified backbone."""
        return Config.BACKBONE_LAYERS.get(backbone_name, None)

    @staticmethod
    def get_device():
        """Get the device for training."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'