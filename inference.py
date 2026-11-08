import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import CustomDefectCNN
from dataset import get_dataloaders

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = get_dataloaders(batch_size=32)

    model = CustomDefectCNN().to(device)
    model.load_state_dict(torch.load("defect_cnn_model.pth", map_location=device, weights_only=True))
    model.eval()

    target_layers = [model.conv1]
    cam = GradCAM(model=model, target_layers=target_layers)

    num_images = 8
    cols = 4
    rows = math.ceil(num_images / cols)

    defective_imgs = []
    defective_lbls = []

    for images, labels in test_loader:
        for i in range(len(labels)):
            if classes[labels[i]] == 'defective':
                defective_imgs.append(images[i:i+1])
                defective_lbls.append(labels[i])
            if len(defective_imgs) == num_images:
                break
        if len(defective_imgs) == num_images:
            break

    _, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i in range(num_images):
        input_tensor = defective_imgs[i].to(device)
        true_label = defective_lbls[i].item()

        targets = [ClassifierOutputTarget(true_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        cam_uint8 = np.uint8(255 * grayscale_cam)

        _, thresh = cv2.threshold(cam_uint8, 140, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        h, w = thresh.shape

        for c in contours:
            area = cv2.contourArea(c)
            if 5 < area < (h * w * 0.15):
                valid_contours.append(c)

        img_np = defective_imgs[i].squeeze(0).permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        img_draw = np.uint8(255 * img_np).copy()

        cv2.drawContours(img_draw, valid_contours, -1, (255, 0, 0), 2)

        ax = axes[i]
        ax.imshow(img_draw)

        with torch.no_grad():
            pred = model(input_tensor).argmax(dim=1).item()

        color = "green" if pred == true_label else "red"
        ax.set_title(f"Valós: {classes[true_label]} \nTipp: {classes[pred]}", color=color)
        ax.axis('off')

    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("batch_evaluation_result.png")
    print("Batch kiértékelés elmentve: batch_evaluation_result.png")

if __name__ == "__main__":
    main()