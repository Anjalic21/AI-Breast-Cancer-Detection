import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class BreastCancerResNet50(nn.Module):
    """ResNet50-based baseline model for Breast Cancer Detection."""
    
    def __init__(self, config):
        """
        Args:
            config (Config): Configuration settings.
        """
        super(BreastCancerResNet50, self).__init__()
        self.config = config
        
        # Load ResNet50 with pretrained ImageNet weights
        print(f"[Model] Initializing ResNet50 with pretrained weights...")
        weights = ResNet50_Weights.DEFAULT if config.PRETRAINED else None
        self.backbone = models.resnet50(weights=weights)
        
        # Extract features input size from the original fully connected layer
        num_features = self.backbone.fc.in_features
        
        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Create a custom classification head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, config.NUM_CLASSES)  # config.NUM_CLASSES is 1 (logits output)
        )
        
    def forward(self, x):
        # Extract features using the ResNet50 backbone
        features = self.backbone(x)
        # Pass features through the custom classifier
        logits = self.classifier(features)
        return logits

    def train(self, mode=True):
        """Overrides the train method to freeze BatchNorm statistics for frozen layers."""
        super(BreastCancerResNet50, self).train(mode)
        
        # If backbone parameters are frozen, enforce correct eval state for BatchNorms
        if hasattr(self, 'backbone'):
            # Check if layer1 parameters are frozen (representing backbone freeze in Stage 1)
            if not next(self.backbone.layer1.parameters()).requires_grad:
                self.backbone.eval()
                
            # If Stage 2 is active (unfreeze layer4, layers 1-3 remain frozen)
            elif not next(self.backbone.layer1.parameters()).requires_grad and next(self.backbone.layer4.parameters()).requires_grad:
                # Force early layers to eval mode
                self.backbone.conv1.eval()
                self.backbone.bn1.eval()
                self.backbone.layer1.eval()
                self.backbone.layer2.eval()
                self.backbone.layer3.eval()
                
                # Force layer4 and classifier to train mode
                self.backbone.layer4.train(mode)
                self.classifier.train(mode)
        return self

    def freeze_backbone(self):
        """Freezes all parameters in the ResNet50 backbone.
        
        Used in Stage 1 of transfer learning.
        """
        print("[Model] Freezing backbone parameters (Stage 1)...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Make sure classifier parameters are trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone_stage2(self):
        """Unfreezes the classifier head and the deep layers (layer4) of the backbone.
        
        Used in Stage 2 of transfer learning to allow fine-tuning of deep feature representation.
        """
        print("[Model] Unfreezing classifier head and Layer 4 of backbone (Stage 2)...")
        
        # Keep initial layers (conv1, bn1, layer1, layer2, layer3) frozen
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 parameters
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
            
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Helper to count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
