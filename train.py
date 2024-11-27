import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import get_dataloaders
from models.tennis_conv import TennisConv, TennisConvResidual, TennisBRNN
from training.config import Config

class Trainer:
    def __init__(self):
        self.writer = SummaryWriter(Config.LOG_DIR)
        self.model = TennisBRNN(Config.NUM_KEYPOINTS, Config.NUM_CLASSES).to(Config.get_device())
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        self.criterion = self._initialize_criterion()
        self.start_epoch, self.best_val_loss = self._load_checkpoint()
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.counter = 0

        json_files = os.listdir("og_dataset/annotations")
        base_path = "og_dataset"
        self.train_loader, self.val_loader, _ = get_dataloaders(json_files, base_path)
        
    def _initialize_optimizer(self):
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY),
            'sgd': optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY),
            'adamw': optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        }
        return optimizers.get(Config.OPTIMIZER)

    def _initialize_scheduler(self):
        schedulers = {
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=Config.LR_SCHEDULER_FACTOR, 
                patience=Config.LR_SCHEDULER_PATIENCE, 
                verbose=True),
            'StepLR': torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=Config.LR_STEP_SIZE, 
                gamma=Config.LR_GAMMA),
            'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=Config.LEARNING_RATE * 10, 
                epochs=Config.EPOCHS, 
                steps_per_epoch=Config.BATCH_SIZE,  # Add this line to specify the number of batches per epoch
                pct_start=0.3, 
                div_factor=25
            ),            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=Config.EPOCHS)
       
        }
        return schedulers.get(Config.LR_SCHEDULER)

    def _initialize_criterion(self):
        return {
            'keypoints': torch.nn.MSELoss(),
            'bbox': torch.nn.MSELoss(),
            'classification': torch.nn.CrossEntropyLoss()
        }

    def _load_checkpoint(self):
        checkpoint_file = f'{Config.CHECKPOINT_DIR}/checkpoint.pth.tar'
        return self._load_checkpoint_from_file(checkpoint_file)

    def _initialize_criterion(self):
        return {
            'keypoints': torch.nn.MSELoss(),
            'bbox': torch.nn.MSELoss(),
            'classification': torch.nn.CrossEntropyLoss()
        }

    def _load_checkpoint_from_file(self, checkpoint_file):
        if os.path.isfile(checkpoint_file):
            print(f"=> Loading checkpoint '{checkpoint_file}'")
            checkpoint = torch.load(checkpoint_file,weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            print(f"=> Loaded checkpoint '{checkpoint_file}' (epoch {start_epoch})")
            return start_epoch, best_val_loss
        else:
            print(f"=> No checkpoint found at '{checkpoint_file}'")
            return 0, float('inf')
        
    def train_model(self):
        print("Starting training...!\n")
        print(f"Using {torch.cuda.get_device_name()}")
        print('-'*70)

        for epoch in range(self.start_epoch, Config.EPOCHS):
            self.model.train()
            running_loss = 0.0
            print(f"Epoch [{epoch + 1}/{Config.EPOCHS}],")
            for images, bboxes, keypoints, labels in tqdm(self.train_loader, desc='Loading training batches'):
                images, bboxes, keypoints, labels = self._move_to_device(images, bboxes, keypoints, labels)
                loss = self._train_step(images, bboxes, keypoints, labels)
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            print(f'Loss: {epoch_loss:.4f}')   

            val_loss = self.validate_model()
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            print(f'Validation Loss after epoch [{epoch + 1}]: {val_loss:.4f}')

            self.scheduler.step(val_loss)
            self._save_checkpoint(epoch, val_loss)

            # if self._check_early_stopping(val_loss):
            #     break

        self.writer.close()

    def _move_to_device(self, images, bboxes, keypoints, labels):
        return (images.to(Config.get_device()), 
                bboxes.to(Config.get_device()), 
                keypoints.to(Config.get_device()), 
                labels.to(Config.get_device()))

    def _train_step(self, images, bboxes, keypoints, labels):
        self.optimizer.zero_grad()
        pred_keypoints, pred_bboxes, pred_classification = self.model(images)
        keypoints = keypoints.view(keypoints.size(0), Config.NUM_KEYPOINTS * 3)

        loss_keypoints = self.criterion['keypoints'](pred_keypoints, keypoints) * Config.LOSS_WEIGHTS['keypoints']
        loss_bbox = self.criterion['bbox'](pred_bboxes, bboxes) * Config.LOSS_WEIGHTS['bbox']
        loss_classification = self.criterion['classification'](pred_classification, labels) * Config.LOSS_WEIGHTS['classification']
        loss = loss_keypoints + loss_bbox + loss_classification

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
        self.optimizer.step()
        return loss

    def validate_model(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, bboxes, keypoints, labels in tqdm(self.val_loader, desc='Loading val set'):
                images, bboxes, keypoints, labels = self._move_to_device(images, bboxes, keypoints, labels)
                pred_keypoints, pred_bboxes, pred_logits = self.model(images)
                keypoints = keypoints.view(keypoints.size(0), Config.NUM_KEYPOINTS * 3)

                loss_keypoints = self.criterion['keypoints'](pred_keypoints, keypoints) * Config.LOSS_WEIGHTS['keypoints']
                loss_bbox = self.criterion['bbox'](pred_bboxes, bboxes) * Config.LOSS_WEIGHTS['bbox']
                loss_classification = self.criterion['classification'](pred_logits, labels) * Config.LOSS_WEIGHTS['classification']

                loss = loss_keypoints + loss_bbox + loss_classification
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        self.best_val_loss = min(val_loss, self.best_val_loss)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
        }
        self._save_checkpoint_to_file(checkpoint, is_best)

    def _save_checkpoint_to_file(self, state, is_best):
        checkpoint_dir = Config.CHECKPOINT_DIR
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filepath = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
        torch.save(state, filepath)
        if is_best:
            best_filepath = os.path.join(checkpoint_dir, 'best_model.pth.tar')
            torch.save(state, best_filepath)

    def _check_early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {self.start_epoch + 1}")
                return True
        return False

def main():
    trainer = Trainer()
    trainer.train_model()

if __name__ == '__main__':
    main()