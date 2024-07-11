import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from Data_Loader_utils import get_data_loaders, write_metrics_to_file
from sklearn.metrics import precision_score, recall_score, accuracy_score
import os

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = DepthwiseSeparableConv(1, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = DepthwiseSeparableConv(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.sa = SpatialAttention()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Adjust according to your final pooled layer size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)  # Apply dropout after pooling
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)  # Apply dropout after pooling
        x = x * self.sa(x)
        #x = self.dropout(x)  # Apply dropout after pooling
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename='checkpoint.pth.tar'):
    return torch.load(filename)


def train_and_validate(data_dir, batch_size, num_epochs, learning_rate, checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    num_classes = len(train_loader.dataset.classes)
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.96)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = 0
    # if checkpoint_path and os.path.isfile(checkpoint_path):
    #     checkpoint = load_checkpoint(checkpoint_path)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch']
    #     scheduler.load_state_dict(checkpoint['scheduler'])

    train_loss, train_precision, train_recall, val_loss, val_precision, val_recall = [], [], [], [], [], []
    train_acc, val_acc = [], []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        all_targets, all_predictions = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        scheduler.step()

        model.eval()
        val_targets, val_predictions = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_targets.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        train_loss.append(total_loss / len(train_loader))
        train_precision.append(precision_score(all_targets, all_predictions, average='macro', zero_division=1))
        train_recall.append(recall_score(all_targets, all_predictions, average='macro'))
        train_acc.append(accuracy_score(all_targets, all_predictions))

        val_loss.append(total_val_loss / len(val_loader))
        val_precision.append(precision_score(val_targets, val_predictions, average='macro', zero_division=1))
        val_recall.append(recall_score(val_targets, val_predictions, average='macro'))
        val_acc.append(accuracy_score(val_targets, val_predictions))

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Train Precision: {train_precision[-1]:.4f}, Train Recall: {train_recall[-1]:.4f}, Train Accuracy: {train_acc[-1]:.4f}')
        print(f'Epoch: {epoch+1}, Val Loss: {val_loss[-1]:.4f}, Val Precision: {val_precision[-1]:.4f}, Val Recall: {val_recall[-1]:.4f}, Val Accuracy: {val_acc[-1]:.4f}')
    write_metrics_to_file(train_loss, train_precision, train_recall, train_acc, val_loss, val_precision, val_recall, val_acc)

    # Save the final model
    save_checkpoint({
        'epoch': num_epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, filename='final_model.pth.tar')


if __name__ == "__main__":
    train_and_validate(r'C:\Users\admin\Desktop\CodeDetection\CNN_Attention\Enhanced_VM_Img', 8, 200, 0.0005, checkpoint_path="./")
