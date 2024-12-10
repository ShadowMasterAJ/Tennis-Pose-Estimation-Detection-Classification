import torch

class Config:
    # General Parameters
    SEED = 42  # For reproducibility

    # Training Parameters
    BATCH_SIZE = 32
    SEQ_LENGTH = 5  # Sequence length for RNN models
    LEARNING_RATE = 0.001
    EPOCHS = 250
    EARLY_STOPPING_PATIENCE = 20  # Early stopping patience
    GRAD_CLIP = 1.0  # Gradient clipping to prevent exploding gradients
    RESUME_TRAINING = True  # If True, resume from checkpoint
    GRAD_ACCUMULATION_STEPS = 4
    IMAGE_SIZE = 320  # Image size for training
    
    # Optimizer Parameters
    OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'adamw'
    MOMENTUM = 0.9  # Momentum factor (for SGD optimizer)
    WEIGHT_DECAY = 1e-4  # L2 regularization (weight decay)
    
    # Learning Rate Scheduler Parameters
    LR_SCHEDULER = 'ReduceLROnPlateau'  # Options: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR', OneCycleLR
    LR_SCHEDULER_PATIENCE = 5  # ReduceLROnPlateau patience before reducing LR
    LR_SCHEDULER_FACTOR = 0.5  # ReduceLROnPlateau LR reduction factor
    LR_STEP_SIZE = 10  # For StepLR, steps before reducing LR
    LR_GAMMA = 0.1  # LR reduction factor for StepLR
    LR_MIN = 1e-6  # Minimum LR for ReduceLROnPlateau and CosineAnnealingLR
    
    
    # Checkpointing Parameters
    LOG_DIR = 'logs_net'
    CHECKPOINT_DIR = 'checkpoints_net'
    SAVE_BEST_ONLY = True  # If True, only save the best-performing model
    SAVE_EVERY_EPOCH = True  # If True, save the model after every epoch
    
    # Regularization Parameters
    DROPOUT_RATE = 0.4 # Dropout rate for model layers
    # Model Architecture Parameters
    NUM_KEYPOINTS = 18  # Number of keypoints to predict
    NUM_CLASSES = 4  # Number of shot classes
    BACKBONE = 'efficientnet_b3'  # Backbone model

    # Model Layer Configurations
    BACKBONE_LAYERS = {
        'efficientnet_b3': {
            'output_channels': 1536,
            'pretrained': True,
            'freeze_layers': True
        },
        'resnet50': {
            'output_channels': 2048,
            'pretrained': True,
            'freeze_layers': True
        },
        'efficientnet_b7': {
            'output_channels': 2560,
            'pretrained': True,
            'freeze_layers': True
        },
        'efficientnet_b0':{
            'output_channels': 1280, 
            'pretrained': True, 
            'freeze_layers': True},
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
