import os
import torch

class Config:
    """Central configuration class for the Breast Cancer Detection pipeline.
    
    Contains hyperparameters, file paths, and device settings.
    """
    def __init__(self):
        # 1. Random seed for reproducibility
        self.SEED = 42
        
        # 2. Device configuration
        if torch.cuda.is_available():
            self.DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            self.DEVICE = "mps"
        else:
            self.DEVICE = "cpu"
            
        # 3. Path configurations
        # We will assume a default datasets folder, which can be overridden
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "datasets", "CBIS-DDSM")
        self.METADATA_CSV = os.path.join(self.DATA_DIR, "metadata.csv")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        
        # Ensure results directory exists
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        # 4. Image preprocessing settings
        # Mammograms are high-resolution; 256x256 is a good compromise for baseline efficiency.
        # ImageNet normalization parameters are standard since we use a pretrained ResNet50.
        self.IMAGE_SIZE = (256, 256)
        self.NORM_MEAN = [0.485, 0.456, 0.406]
        self.NORM_STD = [0.229, 0.224, 0.225]
        
        # 5. Model configurations
        self.MODEL_NAME = "resnet50"
        self.MODEL_TYPE = "baseline"  # 'baseline' or 'multimodal'
        self.ABLATION_MODE = 5  # 2 to 5 for multimodal subsets
        self.PRETRAINED = True
        self.NUM_CLASSES = 1  # 1 output for binary classification (logits for BCEWithLogitsLoss)
        
        # 6. Training parameters
        self.BATCH_SIZE = 16  # Small batch size to avoid OOM errors
        self.CLASS_BALANCING = True  # Address class imbalance dynamically
        
        # Stage 1: Train classifier only (frozen backbone)
        self.STAGE1_EPOCHS = 10
        self.STAGE1_LR = 1e-3
        
        # Stage 2: Fine-tune backbone + classifier (unfrozen layers)
        self.STAGE2_EPOCHS = 20
        self.STAGE2_LR = 1e-5
        
        # Optimizer and Scheduler parameters
        self.WEIGHT_DECAY = 1e-4
        self.EARLY_STOPPING_PATIENCE = 5
        self.LR_PATIENCE = 3
        self.LR_FACTOR = 0.1

    def display(self):
        """Displays the configuration parameters in a readable format."""
        print("=" * 40)
        print("  SYSTEM & PIPELINE CONFIGURATION")
        print("=" * 40)
        for key, value in self.__dict__.items():
            print(f"{key:<25}: {value}")
        print("=" * 40)
