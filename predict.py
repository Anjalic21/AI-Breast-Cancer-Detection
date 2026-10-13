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
from models.multimodal_resnet import BreastCancerMultimodalResNet50
from preprocessing.transforms import get_val_transforms, extract_roi, crop_breast_region
from evaluation.gradcam import GradCAMExplainer


def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Individual Mammogram Inference System")
    parser.add_argument("--image", type=str, required=True, help="Path to input mammogram image")
    parser.add_argument("--mask", type=str, default=None, help="Path to optional binary ROI mask image")
    parser.add_argument("--checkpoint", type=str, default="results/baseline/best_model_stage2.pth", 
                        help="Path to trained model weights checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Classification threshold (default: 0.5)")
    
    # Multimodal parameters
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "multimodal"],
                        help="Model type (baseline or multimodal)")
    parser.add_argument("--density", type=int, choices=[1, 2, 3, 4], help="Breast density (1-4)")
    parser.add_argument("--assessment", type=int, choices=[0, 1, 2, 3, 4, 5], help="BI-RADS assessment score (0-5)")
    parser.add_argument("--subtlety", type=int, choices=[1, 2, 3, 4, 5], help="Subtlety score (1-5)")
    parser.add_argument("--abnormality", type=str, choices=["mass", "calcification"], help="Abnormality type")
    parser.add_argument("--laterality", type=str, choices=["LEFT", "RIGHT"], help="Laterality (LEFT or RIGHT)")
    parser.add_argument("--view", type=str, choices=["CC", "MLO"], help="Image view (CC or MLO)")
    
    args = parser.parse_args()

    # 2. Safety / Research Prototype Disclaimer
    print("=" * 65)
    print("  BREAST CANCER DETECTION SYSTEM — INDIVIDUAL INFERENCE")
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

    # 4. Load configuration & checkpoint
    config = Config()
    
    # Overwrite device setting based on availability (MPS -> CUDA -> CPU)
    if torch.backends.mps.is_available():
        config.DEVICE = "mps"
    elif torch.cuda.is_available():
        config.DEVICE = "cuda"
    else:
        config.DEVICE = "cpu"
        
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(config.BASE_DIR, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file does not exist:\n{checkpoint_path}")
        sys.exit(1)

    print(f"[System] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    state_dict = checkpoint['model_state_dict']
    
    metadata_dim = 0
    meta_slice = []
    meta_tensor = None
    
    if args.model == "multimodal":
        # Inspect state dict to determine metadata dimension
        classifier_weight_key = 'classifier.0.weight'
        if classifier_weight_key in state_dict:
            fused_dim = state_dict[classifier_weight_key].shape[1]
            metadata_dim = fused_dim - 2048
            print(f"[Model] Detected metadata dimension from checkpoint: {metadata_dim}")
        else:
            metadata_dim = 6
            print(f"[Model] Warning: {classifier_weight_key} not found in state dict. Defaulting metadata dimension to 6.")
            
        # Add clear validation errors if required multimodal metadata is missing
        if metadata_dim >= 1 and args.density is None:
            print("ERROR: --density is required for this multimodal model configuration.")
            sys.exit(1)
        if metadata_dim >= 2 and args.assessment is None:
            print("ERROR: --assessment is required for this multimodal model configuration.")
            sys.exit(1)
        if metadata_dim >= 3 and args.subtlety is None:
            print("ERROR: --subtlety is required for this multimodal model configuration.")
            sys.exit(1)
        if metadata_dim >= 6:
            if args.abnormality is None:
                print("ERROR: --abnormality is required for this multimodal model configuration.")
                sys.exit(1)
            if args.laterality is None:
                print("ERROR: --laterality is required for this multimodal model configuration.")
                sys.exit(1)
            if args.view is None:
                print("ERROR: --view is required for this multimodal model configuration.")
                sys.exit(1)
                
        # Encode the metadata using EXACTLY the same encoding/order used during training
        d_val = args.density / 4.0 if args.density is not None else 0.0
        a_val = args.assessment / 5.0 if args.assessment is not None else 0.0
        s_val = args.subtlety / 5.0 if args.subtlety is not None else 0.0
        v_val = 1.0 if args.view is not None and args.view.upper() == 'MLO' else 0.0
        l_val = 1.0 if args.laterality is not None and args.laterality.upper() == 'RIGHT' else 0.0
        ab_val = 1.0 if args.abnormality is not None and args.abnormality.lower() == 'mass' else 0.0
        
        meta_all = [d_val, a_val, s_val, v_val, l_val, ab_val]
        meta_slice = meta_all[:metadata_dim]
        meta_tensor = torch.tensor([meta_slice], dtype=torch.float32).to(config.DEVICE)
        
        # Configure model ablation mode mapping
        if metadata_dim == 1:
            config.ABLATION_MODE = 2
        elif metadata_dim == 2:
            config.ABLATION_MODE = 3
        elif metadata_dim == 3:
            config.ABLATION_MODE = 4
        elif metadata_dim == 6:
            config.ABLATION_MODE = 5
        else:
            config.ABLATION_MODE = 5
            
        model = BreastCancerMultimodalResNet50(config)
    else:
        model = BreastCancerResNet50(config)

    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()

    # If multimodal, monkey patch model's forward to inject meta_tensor when metadata is None (for Grad-CAM)
    if args.model == "multimodal":
        original_forward = model.forward
        def patched_forward(x, metadata=None):
            if metadata is None:
                return original_forward(x, metadata=meta_tensor)
            return original_forward(x, metadata=metadata)
        model.forward = patched_forward

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
    if args.model == "multimodal":
        print(f"[Inference] Running forward pass through Multimodal model with metadata vector: {meta_slice}")
        with torch.no_grad():
            logit = model(img_tensor.to(config.DEVICE), metadata=meta_tensor)
            prob_malignant = torch.sigmoid(logit).item()
    else:
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
        "model": args.model,
        "checkpoint": os.path.basename(checkpoint_path)
    }
    if args.model == "multimodal":
        report_data["metadata_input"] = {
            "density": args.density,
            "assessment": args.assessment,
            "subtlety": args.subtlety,
            "abnormality": args.abnormality,
            "laterality": args.laterality,
            "view": args.view
        }
        report_data["metadata_vector"] = meta_slice

    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=4)

    # 9. Print End-of-Run Summary Report
    print("=" * 65)
    print("  FINAL INFERENCE REPORT")
    print("=" * 65)
    print(f"Checkpoint Loaded Successfully : Yes")
    print(f"Model Type Used                : {args.model}")
    if args.model == "multimodal":
        print(f"Metadata Vector Passed         : {meta_slice}")
    else:
        print(f"Metadata Vector Passed         : N/A (Baseline Image-only)")
    print(f"Malignant Probability          : {prob_malignant * 100:.2f}%")
    print(f"Predicted Class                : {pred_label}")
    print(f"Classification Threshold       : {args.threshold}")
    print(f"Grad-CAM Output Path           : {png_path}")
    print(f"JSON Output Path               : {json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
