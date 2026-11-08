# Industrial Bottle Defect Detection with PyTorch

An end-to-end Computer Vision pipeline built with PyTorch to automate quality control and defect inspection in industrial packaging. The model classifies images of bottles into 'good' or 'defective' categories using a custom Convolutional Neural Network (CNN).

## Features

- **Custom DataLoader Pipeline:** Automated downloading and restructuring of the MVTec-AD dataset.
- **Data Augmentation:** Utilizes `torchvision.transforms` (ColorJitter, RandomFlips, Rotation) to improve model generalization.
- **Class Imbalance Handling:** Implements weighted CrossEntropyLoss to account for the disparity between good and defective samples.
- **Modular Architecture:** Clean separation of dataset preparation, model definition, and training loops.

## Tech Stack

- Python 3.x
- PyTorch & Torchvision
- TorchMetrics
- NumPy & Matplotlib

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
