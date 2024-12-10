import gc
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataloader import get_dataloaders

from models.tennis_brnn import TennisPoseEstimationModel, TennisPoseSPP
from models.tennisnet import TennisNet
from training.config import Config
from torch.amp.autocast_mode import autocast

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Trainer:
    def __init__(self):
        
        json_files = os.listdir("og_dataset/annotations")
        base_path = "og_dataset"
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(json_files, base_path, batch_size=Config.BATCH_SIZE,sequence_length=Config.SEQ_LENGTH)
        
        self.writer = SummaryWriter(Config.LOG_DIR)
        self.model = TennisNet().to(Config.get_device())
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler(self.train_loader)
        self.criterion = self._initialize_criterion()
        self.start_epoch, self.best_val_loss = self._load_checkpoint()
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.counter = 0

    def _initialize_optimizer(self):
        """
        Initializes and returns the optimizer for the model based on the configuration.
        The optimizer is selected from a predefined set of optimizers ('adam', 'sgd', 'adamw')
        using the configuration specified in the Config class.
        Returns:
            torch.optim.Optimizer: The initialized optimizer based on the configuration.
        Raises:
            KeyError: If the optimizer specified in Config.OPTIMIZER is not found in the predefined set.
        """
        
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY),
            'sgd': optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY),
            'adamw': optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        }
        return optimizers.get(Config.OPTIMIZER)

    def _initialize_scheduler(self,train_loader):
        def _initialize_scheduler(self, train_loader):
            """
            Initializes the learning rate scheduler based on the configuration.
            This method sets up different types of learning rate schedulers 
            such as ReduceLROnPlateau, StepLR, OneCycleLR, and CosineAnnealingLR 
            based on the specified configuration in the Config class.
            Parameters:
            - train_loader (DataLoader): The DataLoader for the training dataset, 
              used to determine the number of steps per epoch for the OneCycleLR scheduler.
            Returns:
            - torch.optim.lr_scheduler: The initialized learning rate scheduler 
              as specified in the Config.LR_SCHEDULER.
            """
        
        schedulers = {
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=Config.LR_SCHEDULER_FACTOR, 
                patience=Config.LR_SCHEDULER_PATIENCE, 
                threshold=100
                ),
            'StepLR': torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=Config.LR_STEP_SIZE, 
                gamma=Config.LR_GAMMA),
            'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=Config.LEARNING_RATE * 10, 
                total_steps=len(train_loader) * Config.EPOCHS,
                epochs=Config.EPOCHS, 
                steps_per_epoch=len(train_loader),  # Add this line to specify the number of batches per epoch
                pct_start=0.1, 
                div_factor=10
            ),
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=Config.EPOCHS//2,
                eta_min=Config.LR_MIN
                ),
        }
        return schedulers.get(Config.LR_SCHEDULER)

    def _initialize_criterion(self):
        """
        Initializes and returns a dictionary of loss functions for different tasks.
        The dictionary contains the following key-value pairs:
        - 'keypoints': Mean Squared Error Loss (MSELoss) for keypoint regression.
        - 'bbox': Mean Squared Error Loss (MSELoss) for bounding box regression.
        - 'classification': Cross Entropy Loss (CrossEntropyLoss) for classification tasks.
        Returns:
            dict: A dictionary with keys 'keypoints', 'bbox', and 'classification', 
                  each mapped to their respective loss functions.
        """
        
        return {
            'keypoints': torch.nn.MSELoss(),
            'bbox': torch.nn.MSELoss(),
            'classification': torch.nn.CrossEntropyLoss()
        }

    def _load_checkpoint(self, checkpoint_file=f'{Config.CHECKPOINT_DIR}/best_model.pth.tar'):
        """
        Loads a model checkpoint from a specified file.
        Args:
            checkpoint_file (str): Path to the checkpoint file. Defaults to 
                                   '{Config.CHECKPOINT_DIR}/best_model.pth.tar'.
        Returns:
            tuple: A tuple containing:
                - start_epoch (int): The epoch to start training from.
                - best_val_loss (float): The best validation loss recorded.
        If the checkpoint file exists, the model and optimizer states are loaded 
        from the checkpoint, and the start epoch and best validation loss are 
        returned. If the checkpoint file does not exist, a message is printed and 
        the function returns 0 and infinity for the start epoch and best validation 
        loss, respectively.
        """
        
        if os.path.isfile(checkpoint_file):
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
        
    def _move_to_device(self, images, bboxes, keypoints, labels):
        """
        Moves the given tensors to the specified device.
        Args:
            images (torch.Tensor): The tensor containing image data.
            bboxes (torch.Tensor): The tensor containing bounding box data.
            keypoints (torch.Tensor): The tensor containing keypoint data.
            labels (torch.Tensor): The tensor containing label data.
        Returns:
            tuple: A tuple containing the tensors moved to the specified device.
        """
        
        return (images.to(Config.get_device()), 
                bboxes.to(Config.get_device()), 
                keypoints.to(Config.get_device()), 
                labels.to(Config.get_device()))

    def _check_early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Best validation loss: {:.4f} vs Current validation loss: {:.4f}".format(self.best_val_loss, val_loss))
                print(f"Early stopping at epoch {self.start_epoch + 1}")
                return True
        return False
    
    def print_gpu_memory():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

    def train_model(self):
        """
        Trains the model using the training data loader and evaluates it on the validation data loader.
        This method performs the following steps:
        1. Sets the model to training mode.
        2. Iterates over the epochs specified in the configuration.
        3. For each epoch:
            a. Initializes the running loss.
            b. Prints the current epoch and learning rate.
            c. Iterates over the training data batches:
                i. Moves the data to the appropriate device (CPU/GPU).
                ii. Computes the loss for the current batch.
                iii. Accumulates the running loss.
                iv. Frees up memory by deleting variables and clearing the CUDA cache.
            d. Computes the average loss for the epoch.
            e. Logs the training loss to TensorBoard.
            f. Evaluates the model on the validation data loader and logs the validation loss.
            g. Updates the learning rate scheduler based on the validation loss.
            h. Saves the model checkpoint.
            i. Checks for early stopping and breaks the loop if triggered.
        4. Closes the TensorBoard writer.
        5. Prints a message indicating the completion of training.
        6. Evaluates the model on the test set.
        Note:
            - This method assumes that the model, optimizer, scheduler, data loaders, and other necessary components
              are already initialized and available as attributes of the class instance.
            - The method also assumes that the configuration parameters (e.g., number of epochs, learning rate scheduler type)
              are available in a Config class or similar structure.
        Returns:
            None
        """
        
        print("Starting training...!\n")
        print(f"Using {torch.cuda.get_device_name()}")
        print('-'*70)

        for epoch in range(self.start_epoch, Config.EPOCHS):
            self.model.train()
            running_loss = 0.0
            print(f"Epoch [{epoch + 1}/{Config.EPOCHS}],")
            for param_group in self.optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")

            for images, bboxes, keypoints, labels in tqdm(self.train_loader, desc='Loading training batches'):
                images, bboxes, keypoints, labels = self._move_to_device(images, bboxes, keypoints, labels)
                # print("Expected memory usage: ", images.element_size() * images.nelement() / 1e9)
                # print("Loaded training batch")
                loss = self._train_step(images, bboxes, keypoints, labels)
                running_loss += loss.item()
                # Delete variables to free up memory
                del images, bboxes, keypoints, labels, loss
                
                # Clear CUDA cache and run garbage collector
                torch.cuda.empty_cache()
                gc.collect()

            epoch_loss = running_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            print(f'Epoch [{epoch + 1}/{Config.EPOCHS}] Loss: {epoch_loss:.4f}')   

            val_loss = self.evaluate(data_loader=self.val_loader, mode='Validation')
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            print(f'Validation Loss after epoch [{epoch + 1}]: {val_loss:.4f}')
            
            if Config.LR_SCHEDULER != 'OneCycleLR': 
                self.scheduler.step(val_loss)

            # self.scheduler.step(val_loss)
            self._save_checkpoint(epoch, val_loss)

            if self._check_early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self.writer.close()
        print("Training completed.")
        self.evaluate_test_set()

    def _train_step(self, images, bboxes, keypoints, labels):
        """
        Perform a single training step.
        Args:
            images (torch.Tensor): Batch of input images.
            bboxes (torch.Tensor): Ground truth bounding boxes corresponding to the images.
            keypoints (torch.Tensor): Ground truth keypoints corresponding to the images.
            labels (torch.Tensor): Ground truth labels for classification.
        Returns:
            torch.Tensor: The computed loss for the training step.
        """
    
        self.optimizer.zero_grad()

        with autocast(device_type='cuda'):
            pred_keypoints, pred_bboxes, classification_logits = self.model(images)
                        
            keypoints = keypoints[:, -1, :]  # Flatten across all axes except batch
            bboxes = bboxes[:, -1, :]        # Flatten across all axes except batch
            
            assert keypoints.shape == pred_keypoints.shape, f"Shape mismatch: {keypoints.shape} vs {pred_keypoints.shape}"
            assert bboxes.shape == pred_bboxes.shape, f"Shape mismatch: {bboxes.shape} vs {pred_bboxes.shape}"

            loss_keypoints = self.criterion['keypoints'](pred_keypoints, keypoints) * Config.LOSS_WEIGHTS['keypoints']
            loss_bbox = self.criterion['bbox'](pred_bboxes, bboxes) * Config.LOSS_WEIGHTS['bbox']
            loss_classification = self.criterion['classification'](classification_logits, labels) * Config.LOSS_WEIGHTS['classification']
            
            loss = (loss_keypoints + loss_bbox + loss_classification)
            
        loss.backward()
        clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)
        self.optimizer.step()

        if Config.LR_SCHEDULER == 'OneCycleLR': 
            self.scheduler.step()
        return loss

    def evaluate(self, data_loader, mode='Validation', max_batches=None):
        """
        Evaluate the model on the given data loader.
        Args:
            data_loader (DataLoader): The data loader containing the dataset to evaluate.
            mode (str, optional): The mode of evaluation, either 'Validation' or 'Test'. Defaults to 'Validation'.
            max_batches (int, optional): The maximum number of batches to evaluate. If None, evaluate all batches. Defaults to None.
        Returns:
            float: The average loss over the evaluated batches.
        """
        
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (images, bboxes, keypoints, labels) in enumerate(tqdm(data_loader, desc=f'Loading {mode} batches')):
                if max_batches is not None and i >= max_batches:
                    break
                images, bboxes, keypoints, labels = self._move_to_device(images, bboxes, keypoints, labels)
                
                with autocast(device_type='cuda'):
                    pred_keypoints, pred_bboxes, classification_logits = self.model(images)
                    
                    keypoints = keypoints[:, -1, :]
                    bboxes = bboxes[:, -1, :]
                    
                    loss_keypoints = self.criterion['keypoints'](pred_keypoints, keypoints) * Config.LOSS_WEIGHTS['keypoints']
                    loss_bbox = self.criterion['bbox'](pred_bboxes, bboxes) * Config.LOSS_WEIGHTS['bbox']
                    loss_classification = self.criterion['classification'](classification_logits, labels) * Config.LOSS_WEIGHTS['classification']
        
                    loss = loss_keypoints + loss_bbox + loss_classification
                    running_loss += loss.item()
        
        return running_loss / len(data_loader)
    
    def evaluate_test_set(self):
        """
        Evaluates the model on the test dataset using the best saved model checkpoint.
        This method performs the following steps:
        1. Loads the best model checkpoint from the specified directory.
        2. Loads the model state from the checkpoint.
        3. Evaluates the model on the test dataset.
        4. Prints the test loss.
        5. Logs the test loss to TensorBoard.
        Returns:
            None
        """
        
        # Load the best model
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth.tar')
        if os.path.isfile(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=Config.get_device())
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded best model from {best_model_path}")

        test_loss = self.evaluate(self.test_loader, 'Testing')
        print(f'Test Loss: {test_loss:.4f}')
        self.writer.add_scalar('Loss/test', test_loss)
        
    def _save_checkpoint(self, epoch, val_loss):
        is_best = val_loss < self.best_val_loss
        self.best_val_loss = min(val_loss, self.best_val_loss)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
        }

        checkpoint_dir = Config.CHECKPOINT_DIR
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filepath = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
        torch.save(checkpoint, filepath)
        if is_best:
            best_filepath = os.path.join(checkpoint_dir, 'best_model.pth.tar')
            torch.save(checkpoint, best_filepath)


def main():
    trainer = Trainer()
    
    # Quick test run
    print("Running a quick test run to check if everything works...")

    # Quick training step
    for i, (images, bboxes, keypoints, labels) in enumerate(tqdm(trainer.train_loader, desc='Quick training')):
        if i >= 2:  # Run only for 2 batches
            break
        images, bboxes, keypoints, labels = trainer._move_to_device(images, bboxes, keypoints, labels)
        loss = trainer._train_step(images, bboxes, keypoints, labels)
        print(f"Test run training batch {i + 1} loss: {loss.item():.4f}")
    
    # Quick validation step
    val_loss = trainer.evaluate(data_loader=trainer.val_loader, mode='Validation', max_batches=2)
    print(f"Test run validation loss: {val_loss:.4f}")
    
    # Quick testing step
    test_loss = trainer.evaluate(data_loader=trainer.test_loader, mode='Testing', max_batches=2)
    print(f"Test run test loss: {test_loss:.4f}")
    
    print("Quick test run completed. Starting full training...\n")

    torch.cuda.empty_cache()
    
    trainer.train_model()

    
if __name__ == '__main__':
    main()