import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import MultiScaleDocumentDataset
from src.model import MultiScaleResNet

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

# Data Transformations
train_global_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_patch_tf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_global_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_patch_tf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CSV Files
df_all = pd.read_csv(os.path.join('data', 'real_text_dataset_split.csv'))
test_df = pd.read_csv(os.path.join('data', 'real_text_test_fixed.csv'))

train_df = df_all[df_all['split'] == 'train']
val_df = df_all[df_all['split'] == 'val']

train_loader = DataLoader(
    MultiScaleDocumentDataset(train_df, train_global_tf, train_patch_tf),
    batch_size=16,
    shuffle=True,
    num_workers=2,
)
val_loader = DataLoader(
    MultiScaleDocumentDataset(val_df, val_global_tf, val_patch_tf),
    batch_size=16,
    shuffle=False,
    num_workers=2,
)
test_loader = DataLoader(
    MultiScaleDocumentDataset(test_df, val_global_tf, val_patch_tf),
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

model = MultiScaleResNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

EPOCHS = 5
best_val_acc = 0.0
weights_dir = 'weights'
os.makedirs(weights_dir, exist_ok=True)
best_model_path = os.path.join(weights_dir, 'best_multiscale_fusion.pth')

print('Starting Multi-Scale Fusion training...')

for epoch in range(EPOCHS):
  model.train()
  running_loss, correct, total = 0.0, 0, 0

  for global_imgs, patches, labels in tqdm(
      train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]'
  ):
    global_imgs, patches, labels = (
        global_imgs.to(device),
        patches.to(device),
        labels.to(device),
    )

    optimizer.zero_grad()
    outputs = model(global_imgs, patches)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * global_imgs.size(0)
    _, preds = torch.max(outputs, 1)
    correct += torch.sum(preds == labels.data)
    total += labels.size(0)

  train_acc = (correct.double() / total).item() * 100

  # Validation Phase
  model.eval()
  val_correct, val_total = 0, 0
  with torch.no_grad():
    for global_imgs, patches, labels in val_loader:
      global_imgs, patches, labels = (
          global_imgs.to(device),
          patches.to(device),
          labels.to(device),
      )
      outputs = model(global_imgs, patches)
      _, preds = torch.max(outputs, 1)
      val_correct += torch.sum(preds == labels.data)
      val_total += labels.size(0)

  val_acc = (val_correct.double() / val_total).item() * 100
  print(
      f'Epoch {epoch+1}/{EPOCHS} -> Train Acc: {train_acc:.2f}% | Val Acc:'
      f' {val_acc:.2f}%'
  )

  if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), best_model_path)

# Test Phase
print('Evaluating Proposed Multi-Scale Fusion Model on Test Set...')
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
  for global_imgs, patches, labels in test_loader:
    global_imgs, patches, labels = (
        global_imgs.to(device),
        patches.to(device),
        labels.to(device),
    )
    outputs = model(global_imgs, patches)
    _, preds = torch.max(outputs, 1)
    test_correct += torch.sum(preds == labels.data)
    test_total += labels.size(0)

fusion_test_acc = (test_correct.double() / test_total).item() * 100
print(f'Test Accuracy: {fusion_test_acc:.2f}%')
