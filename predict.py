import argparse
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.dataset import MultiScaleDocumentDataset
from src.model import MultiScaleResNet

LABEL_MAP = {
    0: '0° (Upright)',
    1: '90° (Rotated Clockwise)',
    2: '180° (Upside Down)',
    3: '270° (Rotated Counter-Clockwise)',
}

transform_global = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_patch = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image_path, weights_path, device):
  if not os.path.exists(image_path):
    print(f"Error: Image file not found at '{image_path}'")
    return

  if not os.path.exists(weights_path):
    print(f"Error: Model weights file not found at '{weights_path}'")
    return

  # Load image
  img = Image.open(image_path).convert('RGB')

  # Helper dataset to extract central patches
  dummy_df = pd.DataFrame([{'image_path': image_path, 'label': 0}])
  dataset_helper = MultiScaleDocumentDataset(
      dummy_df, transform_global, transform_patch
  )

  global_tensor, patches_tensor, _ = dataset_helper[0]
  global_tensor = global_tensor.unsqueeze(0).to(device)
  patches_tensor = patches_tensor.unsqueeze(0).to(device)

  # Load Model
  model = MultiScaleResNet(num_classes=4).to(device)
  model.load_state_dict(torch.load(weights_path, map_location=device))
  model.eval()

  # Predict
  with torch.no_grad():
    outputs = model(global_tensor, patches_tensor)
    probs = F.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item() * 100

  print('\n----------------------------------------------')
  print(f'File: {os.path.basename(image_path)}')
  print(f'Predicted Orientation: {LABEL_MAP[pred_class]}')
  print(f'Confidence Score: {confidence:.2f}%')
  print('----------------------------------------------\n')


if __name__ == '__main__':
  import pandas as pd

  parser = argparse.ArgumentParser(
      description='Document Orientation Prediction CLI Tool'
  )
  parser.add_argument(
      '--image',
      '-i',
      type=str,
      required=True,
      help='Path to the input document image',
  )
  parser.add_argument(
      '--weights',
      '-w',
      type=str,
      default='weights/best_multiscale_fusion.pth',
      help='Path to .pth weight file',
  )

  args = parser.parse_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  predict(args.image, args.weights, device)
