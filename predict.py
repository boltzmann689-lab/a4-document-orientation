import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.dataset import MultiScaleDocumentDataset
from src.model import MultiScaleResNet

LABEL_MAP = {
    0: "0° (Upright)",
    1: "90° (Rotated Clockwise)",
    2: "180° (Upside Down)",
    3: "270° (Rotated Counter-Clockwise)",
}

# These transforms MUST match val_global_tf / val_patch_tf in the training script.
# Kaggle: NO ColorJitter, NO RandomRotation (those are augmentation techniques
# used only during training, not during evaluation/prediction).
transform_global = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_patch = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image_path: str, weights_path: str, device: torch.device):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return
    if not os.path.exists(weights_path):
        print(f"Error: Model weights file not found at '{weights_path}'")
        return

    # simulate_capture_pipeline=True: this is the key modification.
    # Raw images on the local machine will be resized to max-640 and compressed
    # with JPEG q85, exactly matching the preprocessing step applied by the
    # Kaggle dataset generation script to EVERY train/val/test image,
    # before central patch extraction.
    dummy_df = pd.DataFrame([{"image_path": image_path, "label": 0}])
    dataset_helper = MultiScaleDocumentDataset(
        dummy_df,
        transform_global,
        transform_patch,
        simulate_capture_pipeline=True,
    )
    global_tensor, patches_tensor, _ = dataset_helper[0]
    global_tensor = global_tensor.unsqueeze(0).to(device)
    patches_tensor = patches_tensor.unsqueeze(0).to(device)

    # Load model - the architecture in src/model.py remains 100% identical
    # to the Kaggle version, so the state_dict can be loaded directly without
    # strict=False.
    model = MultiScaleResNet(num_classes=4).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    # eval() is required so that BatchNorm1d uses the learned running_mean/running_var
    # from training instead of recomputing statistics from the current batch.
    # => batch_size=1 during prediction is completely safe; no need to switch to LayerNorm.
    model.eval()

    with torch.no_grad():
        outputs = model(global_tensor, patches_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item() * 100

    print("\n----------------------------------------------")
    print(f"File: {os.path.basename(image_path)}")
    print(f"Predicted Orientation: {LABEL_MAP[pred_class]}")
    print(f"Confidence Score: {confidence:.2f}%")
    print("----------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Orientation Prediction CLI Tool")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to the input document image")
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default="weights/best_multiscale_fusion.pth",
        help="Path to .pth weight file (copied from Kaggle: best_multiscale_fusion.pth)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(args.image, args.weights, device)