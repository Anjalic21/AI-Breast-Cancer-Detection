# AI Breast Cancer Detection System

A multimodal deep learning system for breast cancer detection using mammography images and clinical metadata, with explainable AI through Grad-CAM and systematic ablation experiments.

## 📌 Project Overview

Breast cancer is one of the most common cancers affecting women worldwide, and early and accurate detection plays an important role in improving clinical outcomes.

This project develops a deep learning-based breast cancer detection system that analyzes mammography images and investigates whether incorporating clinical metadata can improve diagnostic performance.

The system uses a ResNet50-based image backbone and progressively integrates clinical metadata through multimodal fusion. Explainable AI techniques, including Grad-CAM, are incorporated to provide visual insights into the regions influencing model predictions.

## 🎯 Objectives

* Develop a deep learning model for breast cancer classification from mammography images.
* Investigate the effect of clinical metadata on classification performance.
* Compare image-only and multimodal configurations through an ablation study.
* Evaluate the system using multiple clinically relevant performance metrics.
* Provide model interpretability using Grad-CAM visualizations.
* Build a reproducible research-oriented implementation.

## 🧠 System Approach

The system evaluates five experimental configurations:

1. **Image Only — Baseline**
2. **Image + Breast Density**
3. **Image + Breast Density + Assessment**
4. **Image + Breast Density + Assessment + Subtlety**
5. **Image + All Metadata Features — Full Fusion**

The mammography image is processed through a ResNet50-based feature extractor, while available clinical metadata is processed separately and fused with image representations for multimodal prediction.

## 📊 Dataset

The project uses the **CBIS-DDSM (Curated Breast Imaging Subset of the Digital Database for Screening Mammography)** dataset.

The experimental setup uses mammography images together with selected clinical metadata.

> Dataset files are not included in this repository because of their size and dataset distribution considerations.

Users intending to reproduce the experiments should obtain the dataset through its appropriate source and configure the local dataset path accordingly.

## 🔬 Ablation Study

A systematic ablation study was performed to investigate the contribution of individual clinical metadata features.

| Experiment | Features Fused                                 |
| ---------- | ---------------------------------------------- |
| Exp 1      | Image Only                                     |
| Exp 2      | Image + Breast Density                         |
| Exp 3      | Image + Breast Density + Assessment            |
| Exp 4      | Image + Breast Density + Assessment + Subtlety |
| Exp 5      | Image + All Metadata Features                  |

Detailed experimental results are available in:

`ablation_results.md`

## 📈 Evaluation

The system evaluates model performance using:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-Score
* ROC-AUC

The evaluation pipeline is implemented in the `evaluation/` module.

## 🔎 Explainable AI

The project incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize image regions that contribute to the model's prediction.

This provides an additional interpretability layer and helps investigate whether the model is focusing on clinically meaningful regions of mammography images.

Grad-CAM functionality is implemented in:

`evaluation/gradcam.py`

## 🗂️ Project Structure

```text
AI-Breast-Cancer-Detection/
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── gradcam.py
│
├── models/
│   ├── __init__.py
│   ├── multimodal_resnet.py
│   └── resnet_baseline.py
│
├── preprocessing/
│   ├── __init__.py
│   └── transforms.py
│
├── training/
│   ├── __init__.py
│   └── trainer.py
│
├── utils/
│   ├── __init__.py
│   └── config.py
│
├── ablation_results.md
├── main.py
├── predict.py
├── run_ablation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* ResNet50
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Grad-CAM
* Jupyter Notebook

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Anjalic21/AI-Breast-Cancer-Detection.git
cd AI-Breast-Cancer-Detection
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

After configuring the dataset and project paths, the main application can be executed using:

```bash
python main.py
```

For prediction:

```bash
python predict.py
```

For the ablation experiments:

```bash
python run_ablation.py
```

The exact arguments and configuration may vary depending on the dataset path and experimental setup.

## 📁 Results

Generated prediction outputs, model checkpoints, datasets, and other large experimental artifacts are excluded from the Git repository through `.gitignore`.

The repository contains the research code and documentation required to understand and reproduce the experimental pipeline.

## 🔮 Future Work

* Evaluate the system on additional mammography datasets.
* Investigate more advanced multimodal fusion strategies.
* Improve model calibration and robustness.
* Perform external validation on independent datasets.
* Explore additional explainability techniques.
* Investigate deployment in a clinical decision-support environment.

## ⚠️ Disclaimer

This project is developed for academic and research purposes. It is not intended to replace professional medical diagnosis or clinical decision-making.

## 📄 License

License information will be added as the project is prepared for public release.
