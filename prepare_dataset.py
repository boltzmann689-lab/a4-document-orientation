import gc
import os
import random
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SEARCH_DIR = os.path.join('data', 'raw')
OUTPUT_DIR = os.path.join('data', 'real_text_10k_dataset')
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
ROTATION_MAP = {0: 0, 90: 1, 180: 2, 270: 3}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scan raw images
all_raw_images = []
for root, _, files in os.walk(SEARCH_DIR):
  for f in files:
    if f.lower().endswith(VALID_EXTS) and not f.startswith('.'):
      all_raw_images.append(os.path.join(root, f))

print(f'Total raw images found: {len(all_raw_images)}')

TARGET_ORIGINAL_COUNT = 2500
sample_size = min(TARGET_ORIGINAL_COUNT, len(all_raw_images))
selected_raw_images = random.sample(all_raw_images, sample_size)


def rotate_image(image, angle):
  if angle == 0:
    return image
  elif angle == 90:
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
  elif angle == 180:
    return cv2.rotate(image, cv2.ROTATE_180)
  elif angle == 270:
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


records = []
print('Generating rotated 4-angle dataset (10,000 images)...')

for group_id, img_path in enumerate(tqdm(selected_raw_images)):
  img = cv2.imread(img_path)
  if img is None:
    continue

  h, w = img.shape[:2]
  max_dim = 640
  if max(h, w) > max_dim:
    scale = max_dim / float(max(h, w))
    img = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )

  base_name = os.path.basename(img_path)

  for angle, label in ROTATION_MAP.items():
    rotated_img = rotate_image(img, angle)
    new_filename = f'doc_g{group_id}_rot{angle}.jpg'
    save_path = os.path.join(OUTPUT_DIR, new_filename)

    cv2.imwrite(save_path, rotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    records.append({
        'group_id': group_id,
        'image_path': save_path,
        'label': label,
        'angle': angle,
        'original_file': base_name,
    })

df_all = pd.DataFrame(records)

# Stratified split by Group ID
unique_groups = df_all['group_id'].unique()
train_groups, temp_groups = train_test_split(
    unique_groups, test_size=0.20, random_state=SEED
)
val_groups, test_groups = train_test_split(
    temp_groups, test_size=0.50, random_state=SEED
)

df_all['split'] = 'train'
df_all.loc[df_all['group_id'].isin(val_groups), 'split'] = 'val'
df_all.loc[df_all['group_id'].isin(test_groups), 'split'] = 'test'

csv_all_path = os.path.join('data', 'real_text_dataset_split.csv')
csv_test_path = os.path.join('data', 'real_text_test_fixed.csv')

df_all.to_csv(csv_all_path, index=False)
df_all[df_all['split'] == 'test'].to_csv(csv_test_path, index=False)

print('Dataset generation completed successfully.')
