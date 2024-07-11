from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_set = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
def write_metrics_to_file(train_loss, train_precision, train_recall,train_ACC, val_loss, val_precision, val_recall,VALACC):
    with open('metrics.txt', 'w') as f:
        f.write(f"Train Loss: {train_loss}\n")
        f.write(f"Train Precision: {train_precision}\n")
        f.write(f"Train Recall: {train_recall}\n")
        f.write(f"Train ACC: {train_ACC}\n")

        f.write(f"Validation Loss: {val_loss}\n")
        f.write(f"Validation Precision: {val_precision}\n")
        f.write(f"Validation Recall: {val_recall}\n")
        f.write(f"Validation ACC: {VALACC}\n")
