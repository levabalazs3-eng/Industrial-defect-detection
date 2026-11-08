import torch
import torch.nn as nn
from dataset import prepare_data, get_dataloaders
from model import CustomDefectCNN
from trainer import Trainer

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prepare_data(target_dir_name="dataset")
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=32)

    model = CustomDefectCNN().to(device)

    weights = torch.tensor([3.6, 1.0], dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    trainer = Trainer(model=model, classes=classes, loss_fn=loss_fn, lr=1e-3, device=device)
    print("Start of training...")
    history = trainer.fit(train_loader, val_loader, epochs=100)

    torch.save(model.state_dict(), "defect_cnn_model.pth")
    print("The model is sucsefully saved: defect_cnn_model.pth")

if __name__ == "__main__":
    main()