import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.yolo11n_pose import NUM_EPOCHS, CHECKPOINT_INTERVAL

def move_to_device(*tensors, device):
    return [t.to(device) for t in tensors]

def compute_loss(criterion, pred_keypoints, keypoints, pred_bbox, bboxes, pred_classification):
    loss_keypoints = criterion['keypoints'](pred_keypoints, keypoints)
    loss_bbox = criterion['bbox'](pred_bbox, bboxes)
    loss_classification = criterion['classification'](pred_classification)
    return loss_keypoints + loss_bbox + loss_classification

def log_training_progress(writer, epoch, batch_idx, total_batches, loss):
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{total_batches}], "
          f"Loss: {loss.item():.4f}")
    
    global_step = epoch * total_batches + batch_idx
    writer.add_scalar('Training/Total_Loss', loss.item(), global_step)

def log_epoch_summary(writer, epoch, epoch_loss, total_batches):
    avg_epoch_loss = epoch_loss / total_batches
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Loss: {avg_epoch_loss:.4f}")
    writer.add_scalar('Training/Avg_Epoch_Loss', avg_epoch_loss, epoch)
    return avg_epoch_loss

def save_checkpoint(model, optimizer, epoch, avg_epoch_loss):
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = f'checkpoints/{type(model).__name__}_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

def get_loss_criteria():
    return {
        'keypoints': nn.MSELoss(),
        'bbox': nn.MSELoss(),
        'classification': nn.CrossEntropyLoss()
    }