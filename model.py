import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define transformations with data augmentation for training
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
data_dir = 'train'
full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

# Split into train/val/test (80/10/10) - but use provided test if available; here assuming all in data/
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Apply test transform to test set
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model (simple CNN)
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 emotions

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = EmotionCNN()

# Hyperparameter tuning (basic grid search)
learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64]
best_acc = 0
best_params = {}
best_model_state = None

for lr in learning_rates:
    for bs in batch_sizes:
        print(f"Training with lr={lr}, bs={bs}")
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)  # Update batch size
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train
        for epoch in range(10):  # More epochs for better training
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_preds.extend(torch.argmax(outputs, dim=1).tolist())
                val_labels.extend(labels.tolist())
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = {'lr': lr, 'bs': bs}
            best_model_state = model.state_dict()

# Use best model
model.load_state_dict(best_model_state)
print(f"Best params: {best_params}, Val Acc: {best_acc}")

# Evaluation on separate test set
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_preds.extend(torch.argmax(outputs, dim=1).tolist())
        test_labels.extend(labels.tolist())

accuracy = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Per-class metrics (expanded testing)
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(test_labels, test_preds, average=None)
classes = full_dataset.classes
for i, cls in enumerate(classes):
    print(f"{cls}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.savefig('confusion_matrix.png')
plt.close()

# Save model
os.makedirs('trained_models', exist_ok=True)
torch.save(model.state_dict(), 'trained_models/groovy_emotion_detector_v1.pth')
print("Model saved.")