import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def extract_roi(image: np.ndarray, mask: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """Extracts the Region of Interest (ROI) from the image using the binary mask.
    
    Args:
        image (np.ndarray): Grayscale or RGB mammogram image.
        mask (np.ndarray): Binary mask where non-zero pixels represent the lesion.
        margin (float): Extra padding around the bounding box as a fraction of the box size.
        
    Returns:
        np.ndarray: Cropped ROI of the image, or the original image if mask is invalid.
    """
    if mask is None or np.sum(mask) == 0:
        return image
        
    # Ensure mask and image have matching dimensions
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Get coordinates of non-zero elements
    rows, cols = np.where(mask > 0)
    if len(rows) == 0 or len(cols) == 0:
        return image
        
    # Find bounding box
    ymin, ymax = np.min(rows), np.max(rows)
    xmin, xmax = np.min(cols), np.max(cols)
    
    # Calculate dimensions and padding
    height, width = ymax - ymin, xmax - xmin
    pad_h = int(height * margin)
    pad_w = int(width * margin)
    
    # Apply padding with image boundary boundaries check
    ymin = max(0, ymin - pad_h)
    ymax = min(image.shape[0], ymax + pad_h)
    xmin = max(0, xmin - pad_w)
    xmax = min(image.shape[1], xmax + pad_w)
    
    # Crop and return
    return image[ymin:ymax, xmin:xmax]


def crop_breast_region(image: np.ndarray) -> np.ndarray:
    """Segments and crops the breast region to remove large black background areas.
    
    Args:
        image (np.ndarray): Grayscale mammogram image.
        
    Returns:
        np.ndarray: Cropped image containing only the breast tissue.
    """
    # If image is RGB, convert to grayscale for thresholding
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # Otsu thresholding to segment breast from background
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
        
    # Get the largest contour (assumed to be the breast region)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop image
    cropped = image[y:y+h, x:x+w]
    return cropped


def get_train_transforms(image_size: tuple, mean: list, std: list) -> A.Compose:
    """Defines the Albumentations augmentation pipeline for training.
    
    Args:
        image_size (tuple): Target image dimensions (height, width).
        mean (list): Normalization mean.
        std (list): Normalization standard deviation.
        
    Returns:
        A.Compose: Training transformations.
    """
    return A.Compose([
        # Resize to target size
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Spatial Augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=15, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            p=0.5
        ),
        
        # Intensity & Noise Augmentations
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Normalization and PyTorch Tensor conversion
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_val_transforms(image_size: tuple, mean: list, std: list) -> A.Compose:
    """Defines the Albumentations transform pipeline for validation/testing.
    
    Args:
        image_size (tuple): Target image dimensions (height, width).
        mean (list): Normalization mean.
        std (list): Normalization standard deviation.
        
    Returns:
        A.Compose: Validation/testing transformations (no random augmentations).
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
