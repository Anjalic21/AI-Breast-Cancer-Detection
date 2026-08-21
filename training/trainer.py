import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    """Trainer class orchestrating the two-stage training and validation pipeline."""
    
    def __init__(self, model, config, train_loader, val_loader, class_weight=None):
        """
        Args:
            model (nn.Module): The BreastCancerResNet50 model.
            config (Config): Configuration settings.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            class_weight (float): Positional class weight for balancing loss.
        """
        self.model = model.to(config.DEVICE)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weight = class_weight
        self.device = config.DEVICE
        
        # Setup directories for checkpoints
        self.checkpoint_dir = config.RESULTS_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _get_criterion(self):
        """Initializes the binary cross entropy loss, applying class balancing if configured."""
        if self.config.CLASS_BALANCING and self.class_weight is not None:
            # pos_weight should be a tensor of shape [num_classes]
            pos_weight_tensor = torch.tensor([self.class_weight], dtype=torch.float32, device=self.device)
            print(f"[Trainer] Using Class-Balanced BCE Loss (pos_weight: {self.class_weight:.4f})")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            print("[Trainer] Using Standard BCE Loss")
            return nn.BCEWithLogitsLoss()

    def train_epoch(self, optimizer, criterion):
        """Executes a single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                images, metadata, labels = batch
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images, metadata)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
            loss = criterion(logits, labels)
            
            # Backward pass & Optimizer step
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            
            # Binary classification accuracy calculation (logits > 0 corresponds to label 1)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"    [Train] Batch {batch_idx+1:03d}/{num_batches:03d} | Batch Loss: {loss.item():.4f}", end="\r", flush=True)
                
        print()  # Clear line
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    def validate(self, criterion):
        """Evaluates the model on the validation dataset."""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if len(batch) == 3:
                    images, metadata, labels = batch
                    images = images.to(self.device)
                    metadata = metadata.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(images, metadata)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(images)
                loss = criterion(logits, labels)
                
                # Track statistics
                running_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
                # Print batch progress
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    print(f"    [Val] Batch {batch_idx+1:03d}/{num_batches:03d} | Batch Loss: {loss.item():.4f}", end="\r", flush=True)
                    
        print()  # Clear line
        val_loss = running_loss / total_samples
        val_acc = correct_predictions / total_samples
        return val_loss, val_acc

    def save_checkpoint(self, state, filename):
        """Saves a training checkpoint to disk."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        print(f"  --> Checkpoint saved to: {filepath}")

    def load_checkpoint(self, filename):
        """Loads a training checkpoint from disk."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        print(f"[Trainer] Loading model weights from: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def train_stage(self, stage_name, epochs, lr, criterion, unfreeze_func):
        """Handles the training loop for a specific stage (Stage 1 or Stage 2)."""
        print("\n" + "=" * 50)
        print(f" STARTING {stage_name.upper()} TRAINING")
        print("=" * 50)
        
        # Apply layer freeze/unfreeze functions
        unfreeze_func()
        print(f" Trainable Parameters for {stage_name}: {self.model.get_trainable_parameters():,}")
        
        # Setup optimizer and learning rate scheduler
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.config.LR_FACTOR, 
            patience=self.config.LR_PATIENCE
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            val_loss, val_acc = self.validate(criterion)
            
            # Step the learning rate scheduler
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Checkpoint based on Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best state dict
                self.save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'stage': stage_name
                }, f"best_model_{stage_name}.pth")
            else:
                patience_counter += 1
                
            # Early Stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"[Trainer] Early stopping triggered for {stage_name} after {epoch} epochs.")
                break
                
        print(f"Finished {stage_name}. Best Val Loss: {best_val_loss:.4f}")

    def fit(self):
        """Runs the full two-stage training process."""
        criterion = self._get_criterion()
        
        # Stage 1: Warm-up Classifier
        self.train_stage(
            stage_name="stage1",
            epochs=self.config.STAGE1_EPOCHS,
            lr=self.config.STAGE1_LR,
            criterion=criterion,
            unfreeze_func=self.model.freeze_backbone
        )
        
        # Load best weights from Stage 1 before starting Stage 2
        try:
            self.load_checkpoint("best_model_stage1.pth")
        except FileNotFoundError:
            print("[Warning] Best Stage 1 checkpoint not found. Continuing Stage 2 with current weights.")
            
        # Stage 2: Fine-Tuning
        self.train_stage(
            stage_name="stage2",
            epochs=self.config.STAGE2_EPOCHS,
            lr=self.config.STAGE2_LR,
            criterion=criterion,
            unfreeze_func=self.model.unfreeze_backbone_stage2
        )
        
        # Load best overall weights (from Stage 2) for final evaluation
        try:
            self.load_checkpoint("best_model_stage2.pth")
        except FileNotFoundError:
            print("[Warning] Best Stage 2 checkpoint not found. Keeping current weights.")
            
        print("\n[Trainer] Two-stage training pipeline successfully completed.")
