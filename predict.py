import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add current directory to path to ensure relative imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from models.resnet_baseline import BreastCancerResNet50
from preprocessing.transforms import get_val_transforms, extract_roi, crop_breast_region
from evaluation.gradcam import GradCAMExplainer


def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Individual Mammogram Inference System")
    parser.add_argument("--image", type=str, required=True, help="Path to input mammogram image")
    parser.add_argument("--mask", type=str, default=None, help="Path to optional binary ROI mask image")
    parser.add_argument("--checkpoint", type=str, default="results/best_model_stage2.pth", 
                        help="Path to trained model weights checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()

    # 2. Safety / Research Prototype Disclaimer
    print("=" * 65)
    print("  BREAST CANCER DETECTION BASLINE SYSTEM — INDIVIDUAL INFERENCE")
    print("=" * 65)
    print("[Research Safety Notice]")
    print("This output is a research prototype prediction, NOT a medical diagnosis.")
    print("Do not make claims about clinical accuracy from individual images.")
    print("-" * 65)

    # 3. Image & Mask Validation
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"ERROR: Image file does not exist:\n{image_path}")
        sys.exit(1)

    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print(f"ERROR: Failed to load/open image (check path and file integrity):\n{image_path}")
        sys.exit(1)

    mask = None
    if args.mask is not None:
        mask_path = args.mask
        if not os.path.exists(mask_path):
            print(f"ERROR: Mask file does not exist:\n{mask_path}")
            sys.exit(1)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"ERROR: Failed to load/open mask (check path and file integrity):\n{mask_path}")
            sys.exit(1)

    # 4. Load configuration & model architecture
    config = Config()
    
    # Overwrite device setting based on availability (MPS -> CUDA -> CPU)
    if torch.backends.mps.is_available():
        config.DEVICE = "mps"
    elif torch.cuda.is_available():
        config.DEVICE = "cuda"
    else:
        config.DEVICE = "cpu"
        
    model = BreastCancerResNet50(config)

    # Load weights
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(config.BASE_DIR, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file does not exist:\n{checkpoint_path}")
        sys.exit(1)

    print(f"[System] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()

    # 5. Apply inference preprocessing (must match validation flow)
    # Crop breast or extract ROI using mask
    if mask is not None:
        print("[Preprocessing] Segmenting lesion ROI using ground truth mask...")
        cropped_img = extract_roi(original_img, mask, margin=0.1)
    else:
        print("[Preprocessing] Segmenting breast region (auto-removing background)...")
        cropped_img = crop_breast_region(original_img)

    # Convert grayscale cropped image to 3-channel RGB
    preprocessed_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)

    # Apply Albumentations resizing and normalization
    val_transform = get_val_transforms(config.IMAGE_SIZE, config.NORM_MEAN, config.NORM_STD)
    transformed = val_transform(image=preprocessed_img)
    img_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension -> (1, 3, H, W)

    # 6. Run Model Inference
    print("[Inference] Running forward pass through ResNet50 model...")
    with torch.no_grad():
        logit = model(img_tensor.to(config.DEVICE))
        prob_malignant = torch.sigmoid(logit).item()
        
    prob_benign = 1.0 - prob_malignant
    
    # Apply threshold to classify
    pred_label = "MALIGNANT" if prob_malignant >= args.threshold else "BENIGN"
    confidence = prob_malignant if pred_label == "MALIGNANT" else prob_benign

    # Print results to console
    print("-" * 65)
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Prediction: {pred_label}")
    print(f"Malignant Probability: {prob_malignant * 100:.2f}%")
    print(f"Benign Probability: {prob_benign * 100:.2f}%")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("-" * 65)

    # 7. Generate Grad-CAM Explanations
    print("[Grad-CAM] Extracting activation maps from layer4[-1]...")
    explainer = GradCAMExplainer(model, config)
    heatmap = explainer.generate_heatmap(img_tensor)

    # Resize heatmap to match original cropped image dimensions
    heatmap_resized = cv2.resize(heatmap, (cropped_img.shape[1], cropped_img.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)

    # Scale cropped image for overlay blending
    cropped_rgb_float = preprocessed_img.astype(np.float32) / 255.0
    visualization = show_cam_on_image(cropped_rgb_float, heatmap_resized, use_rgb=True)

    # Create jet colormap heatmap representation
    heatmap_jet = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    # 8. Save Results sequentially under results/predictions/
    predictions_dir = os.path.join(config.RESULTS_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    idx = 1
    while True:
        png_filename = f"prediction_{idx:03d}.png"
        json_filename = f"prediction_{idx:03d}.json"
        png_path = os.path.join(predictions_dir, png_filename)
        json_path = os.path.join(predictions_dir, json_filename)
        if not os.path.exists(png_path) and not os.path.exists(json_path):
            break
        idx += 1

    # Save 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # original cropped mammogram
    axes[0].imshow(cropped_rgb_float)
    axes[0].set_title("Cropped Input Mammogram", fontsize=12)
    axes[0].axis("off")
    
    # raw jet color Grad-CAM heatmap
    axes[1].imshow(heatmap_jet)
    axes[1].set_title("Grad-CAM Heatmap (layer4)", fontsize=12)
    axes[1].axis("off")
    
    # blended overlay
    axes[2].imshow(visualization)
    axes[2].set_title(f"Grad-CAM Overlay\n(Malignant Prob: {prob_malignant:.4f})", fontsize=12)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    
    # Save JSON report
    report_data = {
        "image": os.path.abspath(image_path),
        "mask": os.path.abspath(args.mask) if args.mask is not None else None,
        "prediction": pred_label.lower(),
        "malignant_probability": round(prob_malignant, 4),
        "benign_probability": round(prob_benign, 4),
        "confidence": round(confidence, 4),
        "threshold": args.threshold,
        "model": config.MODEL_NAME,
        "checkpoint": os.path.basename(checkpoint_path)
    }

    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=4)

    print(f"Grad-CAM saved to:\n{png_path}")
    print(f"JSON report saved to:\n{json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
