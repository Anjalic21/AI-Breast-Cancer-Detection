import os
import argparse
import random
import ssl

# Disable SSL verification to prevent macOS certificate authority download crashes
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import pandas as pd
import torch

from utils import Config
from preprocessing import get_train_transforms, get_val_transforms
from datasets import get_ddsm_dataloaders
from models import BreastCancerResNet50
from training import Trainer
from evaluation import Evaluator, GradCAMExplainer

def set_seed(seed):
    """Sets random seeds for reproducibility across packages."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior on GPU backends if applicable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[System] Random seed set to: {seed}")


def generate_mock_dataset(config):
    """Generates a synthetic mock CBIS-DDSM dataset for pipeline testing and verification."""
    print("\n" + "=" * 50)
    print(" GENERATING SYNTHETIC CBIS-DDSM DATASET (MOCK)")
    print("=" * 50)
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    images_dir = os.path.join(config.DATA_DIR, "images")
    masks_dir = os.path.join(config.DATA_DIR, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    np.random.seed(42)  # Fixed seed for mock generation
    
    data_rows = []
    
    # Generate 40 synthetic patients (20 train, 10 val, 10 test equivalent)
    # Each patient gets 2 views: CC (Craniocaudal) and MLO (Mediolateral Oblique)
    for p_num in range(1, 41):
        patient_id = f"P_{p_num:05d}"
        pathology = "MALIGNANT" if p_num % 2 == 0 else "BENIGN"
        
        # Determine train/test split designation
        split = "test" if p_num > 30 else "train"
        
        for view in ["CC", "MLO"]:
            img_filename = f"{patient_id}_{view}.png"
            mask_filename = f"{patient_id}_{view}_mask.png"
            
            img_path = os.path.join(images_dir, img_filename)
            mask_path = os.path.join(masks_dir, mask_filename)
            
            # --- 1. Generate Synthetic Mammogram ---
            # Mammograms have a dark background and a brighter, contoured breast region
            img = np.zeros((512, 512), dtype=np.uint8)
            
            # Draw breast contour (semi-ellipse/polygon on the left side)
            pts = np.array([[0, 0], [100, 0], [400, 200], [450, 300], [400, 400], [100, 512], [0, 512]], np.int32)
            cv2.fillPoly(img, [pts], 60)
            
            # Add breast tissue density variations (Gaussian noise and smooth shapes)
            # Create high density fibro-glandular tissue in the center of the breast
            cv2.circle(img, (150, 250), 100, 100, -1)
            cv2.circle(img, (120, 280), 60, 130, -1)
            
            # Add Gaussian blur to simulate breast tissue density distribution
            img = cv2.GaussianBlur(img, (51, 51), 0)
            
            # Add noise to simulate scanner textures
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # --- 2. Generate Synthetic Lesion and Mask ---
            mask = np.zeros((512, 512), dtype=np.uint8)
            
            # We place a simulated lesion inside the breast region (centered around x=150, y=250)
            # Malignant lesions are typically irregular/spiculated, Benign are rounder
            center_x = 150 + np.random.randint(-30, 30)
            center_y = 250 + np.random.randint(-30, 30)
            radius = np.random.randint(15, 30)
            
            if pathology == "MALIGNANT":
                # Draw irregular polygon to simulate spiculated malignant mass
                num_pts = 8
                angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
                points = []
                for angle in angles:
                    r = radius + np.random.randint(-7, 7)
                    px = int(center_x + r * np.cos(angle))
                    py = int(center_y + r * np.sin(angle))
                    points.append([px, py])
                cv2.fillPoly(mask, [np.array(points)], 255)
            else:
                # Draw oval/round shape for benign mass
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
            # Add the lesion shape into the actual mammogram image with high density (brigher)
            img = cv2.addWeighted(img, 1.0, cv2.GaussianBlur(mask, (9, 9), 0), 0.3, 0)
            
            # Save files
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)
            
            # Log references
            data_rows.append({
                "patient_id": patient_id,
                "pathology": pathology,
                "image_path": img_path,
                "mask_path": mask_path,
                "split": split
            })
            
    # Write metadata CSV
    df = pd.DataFrame(data_rows)
    df.to_csv(config.METADATA_CSV, index=False)
    
    print(f"Mock dataset generated successfully at: {config.DATA_DIR}")
    print(f"Total mammogram images: {len(df)}")
    print(f"Metadata file written to: {config.METADATA_CSV}")
    print("=" * 50 + "\n")


def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Breast Cancer Baseline Pipeline (ResNet50)")
    parser.add_argument("--epochs_stage1", type=int, default=None, help="Override Stage 1 epochs")
    parser.add_argument("--epochs_stage2", type=int, default=None, help="Override Stage 2 epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--generate_mock", action="store_true", help="Generate synthetic mock CBIS-DDSM dataset")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and run evaluation only")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "multimodal"], 
                        help="Select model architecture: baseline (default) or multimodal")
    parser.add_argument("--ablation_mode", type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Ablation mode: 1 (Baseline), 2 (Density), 3 (+Assessment), 4 (+Subtlety), 5 (+All)")
    args = parser.parse_args()
    
    # 2. Load settings
    config = Config()
    
    # Apply CLI overrides if set
    if args.epochs_stage1 is not None:
        config.STAGE1_EPOCHS = args.epochs_stage1
    if args.epochs_stage2 is not None:
        config.STAGE2_EPOCHS = args.epochs_stage2
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    config.MODEL_TYPE = args.model
    config.ABLATION_MODE = args.ablation_mode
    
    # If ablation mode is 1, default model type to baseline
    if config.ABLATION_MODE == 1:
        config.MODEL_TYPE = "baseline"
        
    # Route results directory to separate paths to avoid overwriting checkpoints
    if config.MODEL_TYPE == "baseline":
        config.RESULTS_DIR = os.path.join(config.BASE_DIR, "results", "baseline")
    else:
        config.RESULTS_DIR = os.path.join(config.BASE_DIR, "results", f"ablation_{config.ABLATION_MODE}")
        
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
    set_seed(config.SEED)
    config.display()
    
    # 3. Handle Dataset Check / Mock Generation
    # If the CSV metadata or the image paths don't exist, we fallback to generating mock data if flagged
    if args.generate_mock or not os.path.exists(config.METADATA_CSV):
        if not os.path.exists(config.METADATA_CSV):
            print(f"[Warning] CBIS-DDSM metadata CSV not found at: {config.METADATA_CSV}")
            print("Auto-generating a synthetic mock dataset to allow pipeline verification...")
        generate_mock_dataset(config)
        
    # 4. Prepare Transforms & DataLoaders
    train_transform = get_train_transforms(config.IMAGE_SIZE, config.NORM_MEAN, config.NORM_STD)
    val_transform = get_val_transforms(config.IMAGE_SIZE, config.NORM_MEAN, config.NORM_STD)
    
    train_loader, val_loader, test_loader, class_weight = get_ddsm_dataloaders(
        config, train_transform, val_transform
    )
    
    # 5. Instantiate Model
    if config.MODEL_TYPE == "multimodal":
        from models.multimodal_resnet import BreastCancerMultimodalResNet50
        model = BreastCancerMultimodalResNet50(config)
    else:
        model = BreastCancerResNet50(config)
    
    # 6. Training Pipeline (Stages 1 and 2)
    trainer = Trainer(model, config, train_loader, val_loader, class_weight=class_weight)
    
    if not args.skip_training:
        print("[System] Entering training pipeline...")
        trainer.fit()
    else:
        print("[System] Skipping training stage. Loading best Stage 2 weights for evaluation...")
        checkpoint_path = "best_model_stage2.pth"
        try:
            trainer.load_checkpoint(checkpoint_path)
        except FileNotFoundError:
            print(f"[Error] Pre-trained checkpoint '{checkpoint_path}' not found. Cannot evaluate.")
            return
            
    # 7. Evaluation Pipeline
    print("\n[System] Entering evaluation pipeline...")
    evaluator = Evaluator(model, config, test_loader)
    metrics = evaluator.evaluate()
    
    # 8. Explainability Pipeline (Grad-CAM)
    print("\n[System] Generating Grad-CAM visual explanations...")
    explainer = GradCAMExplainer(model, config)
    
    # Let's generate Grad-CAM overlays for a few sample cases from the test dataset
    # We will pick the first 3 samples from the test dataset
    test_df = test_loader.dataset.df
    sample_count = min(3, len(test_df))
    
    for i in range(sample_count):
        row = test_df.iloc[i]
        
        # Load the image and its mask
        img_path = row["image_path"]
        mask_path = row["mask_path"]
        label = row["label"]
        
        # Resolve relative paths in main loop
        if not os.path.isabs(img_path):
            img_path = os.path.join(config.DATA_DIR, img_path)
        if pd.notna(mask_path) and not os.path.isabs(mask_path):
            mask_path = os.path.join(config.DATA_DIR, mask_path)
            
        if not os.path.exists(img_path):
            print(f"[Warning] Image not found for Grad-CAM: {img_path}. Skipping.")
            continue
            
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"[Warning] Failed to load image for Grad-CAM: {img_path}. Skipping.")
            continue
            
        mask = None
        if pd.notna(mask_path) and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
        # Apply validation preprocessing to get the single input tensor
        preprocessed_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        transformed = val_transform(image=preprocessed_img)
        img_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension -> (1, 3, H, W)
        
        # Get metadata features for this specific test case matching the ablation order
        left_right_val = 1 if str(row.get('left_or_right', 'LEFT')).upper() == 'RIGHT' else 0
        view_val = 1 if str(row.get('image_view', 'MLO')).upper() == 'MLO' else 0
        abnormality_val = 1 if str(row.get('abnormality_type', 'mass')).lower() == 'mass' else 0
        
        density_val = float(row.get('breast_density', 2)) / 4.0
        assessment_val = float(row.get('assessment', 0)) / 5.0
        subtlety_val = float(row.get('subtlety', 0)) / 5.0
        
        meta_arr = np.array([
            density_val, assessment_val, subtlety_val,
            view_val, left_right_val, abnormality_val
        ], dtype=np.float32)
        
        # Determine dynamic metadata dimension for this ablation mode
        meta_dim = 6
        if hasattr(config, 'ABLATION_MODE'):
            if config.ABLATION_MODE == 2:
                meta_dim = 1
            elif config.ABLATION_MODE == 3:
                meta_dim = 2
            elif config.ABLATION_MODE == 4:
                meta_dim = 3
            elif config.ABLATION_MODE == 5:
                meta_dim = 6
                
        meta_slice = meta_arr[:meta_dim]
        metadata_tensor = torch.tensor(meta_slice).unsqueeze(0).to(config.DEVICE)  # Shape (1, meta_dim)

        # Get model prediction probability
        model.eval()
        with torch.no_grad():
            if config.MODEL_TYPE == "multimodal":
                logit = model(img_tensor.to(config.DEVICE), metadata_tensor)
            else:
                logit = model(img_tensor.to(config.DEVICE))
            prob = torch.sigmoid(logit).item()
            
        # Generate plot
        filename = f"gradcam_explanation_sample_{i+1}.png"
        explainer.save_explanation_plot(
            original_image=original_img,
            input_tensor=img_tensor,
            label=label,
            prob=prob,
            mask=mask,
            filename=filename
        )
        
    print("\n" + "=" * 50)
    print(" PIPELINE EXECUTION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
