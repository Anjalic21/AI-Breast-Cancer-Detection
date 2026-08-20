import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAMExplainer:
    """Class to generate and save Grad-CAM explanation visualizations for the ResNet50 model."""
    
    def __init__(self, model, config):
        """
        Args:
            model (nn.Module): The trained BreastCancerResNet50 model.
            config (Config): Configuration settings.
        """
        self.model = model.eval()
        self.config = config
        self.device = config.DEVICE
        
        # In ResNet50, layer4[-1] represents the final convolutional layer of the backbone.
        # This is where the spatial context is combined with high-level feature representations.
        self.target_layers = [self.model.backbone.layer4[-1]]
        
        # Instantiate the GradCAM generator (automatically runs on model's device)
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, input_tensor, target_category=0):
        """Generates raw Grad-CAM activation map for a given input tensor.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor of shape (1, 3, H, W).
            target_category (int): Target index for classification output (0 for binary class).
            
        Returns:
            np.ndarray: Calculated 2D activation heatmap normalized to [0, 1].
        """
        # Ensure tensor is on the correct device and in evaluation mode
        input_tensor = input_tensor.to(self.device)
        
        # ClassifierOutputTarget(0) maps the gradients to the final binary classification neuron output.
        targets = [ClassifierOutputTarget(target_category)]
        
        # Generate raw activation map
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]

    def save_explanation_plot(self, original_image, input_tensor, label, prob, mask=None, filename="gradcam_explanation.png"):
        """Generates a side-by-side visualization plot: Original, Mask (optional), and Grad-CAM overlay.
        
        Args:
            original_image (np.ndarray): Original raw/processed image of shape (H, W, 3).
                                         Values should be in range [0, 255] or [0, 1].
            input_tensor (torch.Tensor): Preprocessed input tensor of shape (1, 3, H, W).
            label (int): Ground truth label (0 for Benign, 1 for Malignant).
            prob (float): Model's predicted probability of malignancy.
            mask (np.ndarray): Binary segmentation mask (optional) of shape (H, W).
            filename (str): Name of output plot image file.
        """
        # Ensure original image is normalized to [0, 1] for overlay blending
        if original_image.max() > 1.0:
            rgb_img = original_image.astype(np.float32) / 255.0
        else:
            rgb_img = original_image.copy()
            
        # Ensure it is 3-channel
        if len(rgb_img.shape) == 2:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
            
        # Generate heatmaps
        heatmap = self.generate_heatmap(input_tensor)
        
        # Resize heatmap to match the original image dimensions (width, height)
        heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Overlay the activation map on the original image
        visualization = show_cam_on_image(rgb_img, heatmap_resized, use_rgb=True)
        
        # Setup subplot layout based on mask availability
        num_cols = 3 if mask is not None else 2
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
        
        # Map label names
        label_map = {0: "Benign", 1: "Malignant"}
        
        # 1. Plot original image
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original Mammogram\n(GT: {label_map[int(label)]})", fontsize=12)
        axes[0].axis("off")
        
        # 2. Plot mask if available
        col_idx = 1
        if mask is not None:
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Ground Truth Mask\n(Lesion ROI)", fontsize=12)
            axes[1].axis("off")
            col_idx = 2
            
        # 3. Plot Grad-CAM overlay
        axes[col_idx].imshow(visualization)
        axes[col_idx].set_title(f"Grad-CAM Explanation\n(Pred Prob: {prob:.4f})", fontsize=12)
        axes[col_idx].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(self.config.RESULTS_DIR, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Grad-CAM] Explanation visualization saved to: {save_path}")
        return save_path
