import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Dataset class with on-the-fly image rotation
class DocumentOrientationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.rotations = [0, 90, 180, 270]

    def __len__(self):
        return len(self.df) * 4

    def __getitem__(self, idx):
        img_idx = idx // 4
        rot_idx = idx % 4
        
        img_path = self.df.iloc[img_idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        
        angle = self.rotations[rot_idx]
        if angle != 0:
            image = image.rotate(angle, expand=True)
            
        if self.transform:
            image = self.transform(image)
            
        return image, rot_idx

# Image preprocessing and normalization
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize DataLoaders
train_dataset = DocumentOrientationDataset("train.csv", transform=data_transforms)
val_dataset   = DocumentOrientationDataset("val.csv", transform=data_transforms)
test_dataset  = DocumentOrientationDataset("test.csv", transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"DataLoaders initialized: Train ({len(train_dataset)}) | Val ({len(val_dataset)}) | Test ({len(test_dataset)})")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Model setup (ResNet18)
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 10
best_val_acc = 0.0

print("\nStarting Training Loop...\n" + "-"*50)

for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data)
        total_train += inputs.size(0)
        
    train_loss = running_loss / total_train
    train_acc = correct_train.double() / total_train
    
    # Validation phase
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data)
            total_val += inputs.size(0)
            
    val_loss = val_loss / total_val
    val_acc = correct_val.double() / total_val
    
    print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
          f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # Save best model checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_rotnet_model.pth")
        print(f"   --> Saved Best Model Checkpoint (Val Acc: {best_val_acc:.4f})")

print("\nTraining complete!")