# Industrial Bottle Defect Detection with PyTorch

An end-to-end Computer Vision pipeline built with PyTorch to automate quality control and defect inspection in industrial packaging. The model classifies images of bottles into 'good' or 'defective' categories using a custom Convolutional Neural Network (CNN).

## Features

- **Inference REST API:** Production-ready backend built with FastAPI to serve model predictions via HTTP requests (includes Swagger UI).
- **Explainable AI & Defect Localization:** Uses PyTorch Grad-CAM combined with OpenCV contour detection to isolate and draw bounding boxes around structural anomalies.
- **Custom DataLoader Pipeline:** Automated downloading and restructuring of the MVTec-AD dataset.
- **Data Augmentation:** Utilizes `torchvision.transforms` (ColorJitter, RandomFlips, Rotation) to improve model generalization.
- **Class Imbalance Handling:** Implements weighted CrossEntropyLoss to account for the disparity between good and defective samples.
- **Modular Architecture:** Clean separation of dataset preparation, model definition, and training loops.

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision, TorchMetrics, Torchsummary
- **Computer Vision:** OpenCV, Grad-CAM, Pillow
- **API & Serving:** FastAPI, Uvicorn
- **Data & Visualization:** NumPy, Matplotlib, Kagglehub

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   and if you are a windows user also run this.

   ```bash
   set KMP_DUPLICATE_LIB_OK=True
   ```

2. Train the Model

   Downloads the dataset, trains the CNN, and saves the weights(`defect_cnn_model.pth`).

   ```bash
   python main.py
   ```

3. Batch Evaluation & Localization
   Evaluates the test set and visually isolates defects using Grad-CAM and OpenCV area-filtering. Saves the result as `batch_evaluation_result.png`.

   ```bash
   python inference.py
   ```

4. Run the REST API
   Start the FastAPI server production-like interface
   ```bash
   python api.py
   ```
   Once runnung open `http://localhost:8000/docs` in your browser to test the endpoints via interactive Swagger UI.
