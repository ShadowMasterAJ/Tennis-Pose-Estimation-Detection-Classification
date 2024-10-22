import torch
import torch.optim as optim
from training.config import Config
from torch.utils.tensorboard import SummaryWriter
import os
from dataloader import get_dataloaders
from models.tennis_conv import TennisConv
from tqdm import tqdm

def train_model():
    print("Starting training...!\n")
    # Initialize TensorBoard
    writer = SummaryWriter(Config.LOG_DIR)

    json_files = os.listdir("og_dataset\\annotations")
    base_path = "og_dataset"
    train_loader, val_loader, _ = get_dataloaders(json_files, base_path)
    for images, bboxes, keypoints, labels in train_loader:
        print(images.shape, bboxes.shape, keypoints.shape, labels.shape)
        break
    # Model, Loss, Optimizer
    model = TennisConv(Config.NUM_KEYPOINTS, Config.NUM_CLASSES).cuda()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Loss Functions
    criterion = {
        'keypoints': torch.nn.MSELoss(),            # For keypoint heatmaps
        'bbox': torch.nn.MSELoss(),                 # For bounding boxes
        'classification': torch.nn.CrossEntropyLoss()  # For shot classification
    }

    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for images, bboxes, keypoints, labels in tqdm(train_loader):
            images = images.to(Config.get_device())
            bboxes = bboxes.to(Config.get_device())
            keypoints = keypoints.to(Config.get_device())
            labels = labels.to(Config.get_device())

            optimizer.zero_grad()

            pred_keypoints, pred_bboxes, pred_classification = model(images)

            # Reshape keypoints to (batch_size, 54) -> each image has 18 keypoints with (x, y, v)
            keypoints = keypoints.view(keypoints.size(0), Config.NUM_KEYPOINTS * 3)

            # Compute losses
            loss_keypoints = criterion['keypoints'](pred_keypoints, keypoints) * Config.LOSS_WEIGHTS['keypoints']
            loss_bbox = criterion['bbox'](pred_bboxes, bboxes) * Config.LOSS_WEIGHTS['bbox']
            loss_classification = criterion['classification'](pred_classification, labels) * Config.LOSS_WEIGHTS['classification']

            # Total loss
            loss = loss_keypoints + loss_bbox + loss_classification
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        print(f'Epoch [{epoch + 1}/{Config.EPOCHS}], Loss: {epoch_loss:.4f}')

    writer.close()
    
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, bboxes, keypoints, labels in val_loader:
            images, bboxes, keypoints, labels = images.cuda(), bboxes.cuda(), keypoints.cuda(), labels.cuda()

            pred_keypoints, pred_bboxes, pred_logits = model(images)

            # Compute validation losses
            loss_keypoints = criterion['keypoints'](pred_keypoints, keypoints)
            loss_bbox = criterion['bbox'](pred_bboxes, bboxes)
            loss_classification = criterion['classification'](pred_logits, labels)

            loss = loss_keypoints + loss_bbox + loss_classification
            val_loss += loss.item()

    return val_loss / len(val_loader)

if __name__ == '__main__':
    train_model()
