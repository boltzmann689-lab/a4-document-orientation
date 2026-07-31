import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# Dataset class definition
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

# Target labels
target_names = ['0° (Upright)', '90° (Clockwise)', '180° (Upside-down)', '270° (Counter-clockwise)']

# Data preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize DataLoaders
train_dataset = DocumentOrientationDataset("train.csv", transform=data_transforms)
val_dataset   = DocumentOrientationDataset("val.csv", transform=data_transforms)
test_dataset  = DocumentOrientationDataset("test.csv", transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model_path = "best_rotnet_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model weights from '{model_path}'")
else:
    print(f"Warning: '{model_path}' not found! Evaluating on un-trained model.")

model = model.to(device)

# Evaluation function
def evaluate_and_get_cm(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    return cm, report_dict, report_str

# Evaluate all datasets
cm_train, rep_train_dict, rep_train_str = evaluate_and_get_cm(train_loader)
cm_val,   rep_val_dict,   rep_val_str   = evaluate_and_get_cm(val_loader)
cm_test,  rep_test_dict,  rep_test_str  = evaluate_and_get_cm(test_loader)

# Print Classification Reports
print("="*65 + "\nTRAIN SET CLASSIFICATION REPORT:\n" + "="*65)
print(rep_train_str)

print("="*65 + "\nVALIDATION SET CLASSIFICATION REPORT:\n" + "="*65)
print(rep_val_str)

print("="*65 + "\nTEST SET CLASSIFICATION REPORT:\n" + "="*65)
print(rep_test_str)

# Plot and save Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=target_names, yticklabels=target_names)
axes[0].set_title('Train Set Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True Label')

sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=target_names, yticklabels=target_names)
axes[1].set_title('Validation Set Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True Label')

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
            xticklabels=target_names, yticklabels=target_names)
axes[2].set_title('Test Set Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True Label')

plt.tight_layout()
plt.savefig("confusion_matrices_all.png", dpi=300)
plt.show()

# Print Performance Summary Table
summary_df = pd.DataFrame({
    'Dataset Split': ['Train Set (70%)', 'Validation Set (15%)', 'Test Set (15%)'],
    'Accuracy': [f"{rep_train_dict['accuracy']*100:.2f}%", 
                 f"{rep_val_dict['accuracy']*100:.2f}%", 
                 f"{rep_test_dict['accuracy']*100:.2f}%"],
    'Macro F1-Score': [f"{rep_train_dict['macro avg']['f1-score']:.4f}", 
                       f"{rep_val_dict['macro avg']['f1-score']:.4f}", 
                       f"{rep_test_dict['macro avg']['f1-score']:.4f}"]
})

print("\n" + "="*50)
print("EVALUATION PERFORMANCE SUMMARY")
print("="*50)
print(summary_df.to_string(index=False))
print("="*50)