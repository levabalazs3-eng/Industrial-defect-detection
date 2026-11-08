import shutil
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import kagglehub

def prepare_data(target_dir_name: str = "dataset") -> Path:
    path = kagglehub.dataset_download("ipythonx/mvtec-ad")
    source_dir = Path(path) / "bottle"
    target_dir = Path("dataset")

    good_dir = target_dir / "good"
    defective_dir = target_dir / "defective"

    good_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)

    def copy_images(src_folder, dest_folder):
        if not src_folder.exists(): return
        for file in src_folder.iterdir():
            if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                new_filename = f"{src_folder.parent.name}_{src_folder.name}_{file.name}"
                shutil.copy(file, dest_folder / new_filename)

    print("Refectoring images...")

    copy_images(source_dir / "train" / "good", good_dir)
    copy_images(source_dir / "test" / "good", good_dir)

    test_dir = source_dir / "test"
    if test_dir.exists():
        for defect_folder in test_dir.iterdir():
            if defect_folder.is_dir() and defect_folder.name != "good":
                copy_images(defect_folder, defective_dir)

    print(f"Number of good images: {len(list(good_dir.iterdir()))}")
    print(f"Number of bad images: {len(list(defective_dir.iterdir()))}")

class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

def get_dataloaders(data_dir: str = "dataset", batch_size: int = 32, seed: int = 23):
    train_transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    classes = full_dataset.classes

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set,val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(23)
    )
    train_dataset = TransformDataset(train_set, train_transform)
    val_dataset = TransformDataset(val_set, val_transform)
    test_dataset = TransformDataset(test_set, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, classes