import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)

class Evaluator:
    """Evaluator class for computing performance metrics and plotting visualization curves."""
    
    def __init__(self, model, config, test_loader):
        """
        Args:
            model (nn.Module): The trained BreastCancerResNet50 model.
            config (Config): Configuration settings.
            test_loader (DataLoader): DataLoader for testing.
        """
        self.model = model.to(config.DEVICE)
        self.config = config
        self.test_loader = test_loader
        self.device = config.DEVICE
        self.results_dir = config.RESULTS_DIR
        
    def predict(self):
        """Runs model inference on test dataset and collects labels and predictions.
        
        Returns:
            tuple: (numpy arrays of ground truth labels, predicted probabilities, and binary predictions)
        """
        self.model.eval()
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                if len(batch) == 3:
                    images, metadata, labels = batch
                    images = images.to(self.device)
                    metadata = metadata.to(self.device)
                    logits = self.model(images, metadata)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    logits = self.model(images)
                    
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        return (
            np.array(all_labels).flatten(),
            np.array(all_probs).flatten(),
            np.array(all_preds).flatten()
        )

    def calculate_metrics(self, labels, probs, preds):
        """Computes standard classification evaluation metrics.
        
        Returns:
            dict: Performance metrics values.
        """
        # Calculate scores
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall (sensitivity)": recall,
            "f1_score": f1,
            "auc": roc_auc
        }
        
        # Display results
        print("\n" + "=" * 50)
        print(" PERFORMANCE METRICS REPORT")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"  {key.capitalize():<22}: {value:.4f}")
        print("-" * 50)
        print("Classification Report:")
        print(classification_report(labels, preds, target_names=["Benign", "Malignant"], zero_division=0))
        print("=" * 50)
        
        return metrics

    def plot_confusion_matrix(self, labels, preds, filename="confusion_matrix.png"):
        """Plots and saves the Confusion Matrix heatmap."""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=["Benign", "Malignant"], 
            yticklabels=["Benign", "Malignant"],
            cbar=False
        )
        
        plt.title("Confusion Matrix", fontsize=14, pad=15)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Evaluator] Confusion Matrix plot saved to: {save_path}")

    def plot_roc_curve(self, labels, probs, filename="roc_curve.png"):
        """Plots and saves the ROC curve."""
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Evaluator] ROC Curve plot saved to: {save_path}")

    def evaluate(self):
        """Runs the complete evaluation suite."""
        print("\n[Evaluator] Running model predictions on evaluation dataset...")
        labels, probs, preds = self.predict()
        
        metrics = self.calculate_metrics(labels, probs, preds)
        self.plot_confusion_matrix(labels, preds)
        self.plot_roc_curve(labels, probs)
        
        return metrics
