import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Data_Loader_utils import get_data_loaders, write_metrics_to_file
from sklearn.metrics import precision_score, recall_score, accuracy_score
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=11, padding=5)
        self.fc1 = nn.Linear(4 * 128 * 128, 128)  # Updated size calculation
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))  # Inefficient activation
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.fc1(x))  # No non-linearity here
        x = self.fc2(x)
        return x

def train_and_validate(data_dir, batch_size, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    num_classes = len(train_loader.dataset.classes)
    model = CNN(num_classes).to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss, train_precision, train_recall, val_loss, val_precision, val_recall = [], [], [], [], [], []
    Train_ACC,Val_ACC = [],[]
    for epoch in range(num_epochs):
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
        train_acc = accuracy_score(all_targets, all_predictions)
        Train_ACC.append(train_acc)
        val_loss.append(total_val_loss / len(val_loader))
        val_precision.append(precision_score(val_targets, val_predictions, average='macro', zero_division=1))
        val_recall.append(recall_score(val_targets, val_predictions, average='macro'))
        val_accuracy = accuracy_score(val_targets, val_predictions)
        Val_ACC.append(val_accuracy)
        #print(f'Epoch: {epoch}, Train Loss: {train_loss}, Train ACC: {train_acc},  Train Precision: {train_precision}, Train Recall: {train_recall}')
        #print(f'Epoch: {epoch}, Val Loss: {val_loss}, VAL ACC: {val_accuracy},Val Precision: {val_precision}, Val Recall: {val_recall}')

    write_metrics_to_file(train_loss, train_precision, train_recall,Train_ACC, val_loss, val_precision, val_recall,Val_ACC)

if __name__ == "__main__":
    train_and_validate(r'C:\Users\admin\Desktop\CodeDetection\CNN_Attention\Enhanced_Org_Img', 16, 200, 0.0005)
