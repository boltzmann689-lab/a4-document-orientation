import os
import shutil
import glob
import random
import pandas as pd
from PIL import Image

# Initialize directory structure
base_dir = "data_1000"
sub_categories = ['scan', 'receipt', 'handwritten', 'captured']

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

for sub in sub_categories:
    os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install -q datasets")
    from datasets import load_dataset

# 1. Download Scan Group (DocVQA)
scan_dir = os.path.join(base_dir, 'scan')
count_scan = 0
ds_cord_test = load_dataset("nielsr/docvqa_1200_examples", split="train", streaming=True)

for item in ds_cord_test:
    if count_scan >= 250:
        break
    img = item['image'].convert("RGB")
    img.save(os.path.join(scan_dir, f"scan_{count_scan+1:04d}.jpg"))
    count_scan += 1

# 2. Download Receipt Group (CORD)
receipt_dir = os.path.join(base_dir, 'receipt')
ds_cord_train = load_dataset("naver-clova-ix/cord-v2", split="train")
count_receipt = 0

for idx, item in enumerate(ds_cord_train):
    if count_receipt >= 250:
        break
    img = item['image'].convert("RGB")
    img.save(os.path.join(receipt_dir, f"receipt_{count_receipt+1:04d}.jpg"))
    count_receipt += 1

# 3. Process Handwritten Group (Local Kaggle Input #1: IAM Forms)
hw_dir = os.path.join(base_dir, 'handwritten')
count_hw = 0
local_iam_path = "/kaggle/input/datasets/naderabdelghany/iam-handwritten-forms-dataset"

if os.path.exists(local_iam_path):
    for root, _, files in os.walk(local_iam_path):
        for file in files:
            if count_hw >= 250:
                break
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    # Filter for full-page A4 documents
                    if img.width > 400 and img.height > 400:
                        img.save(os.path.join(hw_dir, f"handwritten_{count_hw+1:04d}.jpg"))
                        count_hw += 1
                except Exception:
                    continue
        if count_hw >= 250:
            break
else:
    print(f"Warning: Local dataset path not found: {local_iam_path}")

# 4. Process Captured Group (Local Kaggle Input #2: SmartDoc)
captured_dir = os.path.join(base_dir, "captured")
count_cap = 0
local_smartdoc_path = "/kaggle/input/datasets/octaviusgaster/smartdoc2015-extracted-frames/smart_doc_extracted/images"

if os.path.exists(local_smartdoc_path):
    for root, _, files in os.walk(local_smartdoc_path):
        for file in files:
            if count_cap >= 250:
                break
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.save(os.path.join(captured_dir, f"captured_{count_cap+1:04d}.jpg"))
                    count_cap += 1
                except Exception:
                    continue
        if count_cap >= 250:
            break
else:
    print(f"Warning: Local dataset path not found: {local_smartdoc_path}")

# Dataset Summary
print("\n" + "="*40)
print("DATASET SUMMARY")
print("="*40)
total_images = 0
for sub in sub_categories:
    sub_path = os.path.join(base_dir, sub)
    num_imgs = len(os.listdir(sub_path))
    total_images += num_imgs
    print(f"Category [{sub:<11}]: {num_imgs:>4} images")

print("-" * 40)
print(f"TOTAL               : {total_images:>4} images")
print("="*40)