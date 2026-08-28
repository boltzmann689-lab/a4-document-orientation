import argparse
import os
import re
import subprocess
import sys
from PIL import Image

try:
  import kagglehub
except ImportError:
  subprocess.run(
      [sys.executable, '-m', 'pip', 'install', 'kagglehub', 'pillow'],
      check=True,
  )
  import kagglehub

LABEL_MAP = {
    0: '0° (Upright)',
    1: '90° (Rotated Clockwise)',
    2: '180° (Upside Down)',
    3: '270° (Rotated Counter-Clockwise)',
}

DATASET_CATEGORY_MAP = {
    'urbikn/sroie-datasetv2': 'receipt',
    'lonelvino/cord-1000': 'receipt',
    'dhiaznaidi/receiptdatasetssd300v2': 'receipt',
}

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


def check_kaggle_credentials():
  if not (
      (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))
      or os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json'))
  ):
    sys.exit(1)


def find_images_in_dir(root_dir):
  return [
      os.path.join(dp, f)
      for dp, _, filenames in os.walk(root_dir)
      for f in filenames
      if f.lower().endswith(IMAGE_EXTS)
  ]


def collect_real_images(target_count=100):
  check_kaggle_credentials()
  receipts = []

  for slug in DATASET_CATEGORY_MAP:
    if len(receipts) >= target_count:
      break
    try:
      local_path = kagglehub.dataset_download(slug)
    except Exception:
      continue

    for p in find_images_in_dir(local_path):
      if len(receipts) >= target_count:
        break
      try:
        receipts.append(Image.open(p).convert('RGB'))
      except Exception:
        continue

  return {'receipt': receipts}


def prepare_real_kaggle_dataset(base_dir, num_base=100):
  cat_dir = os.path.join(base_dir, 'receipt')
  existing = (
      [f for f in os.listdir(cat_dir) if f.endswith('.jpg')]
      if os.path.exists(cat_dir)
      else []
  )
  if len(existing) >= num_base * 4:
    return

  real_images = collect_real_images(num_base).get('receipt', [])
  if not real_images:
    return

  os.makedirs(cat_dir, exist_ok=True)
  for idx, img in enumerate(real_images, 1):
    img = img.copy()
    img.thumbnail((1200, 1200))
    idx_str = f'{idx:03d}'
    img.save(os.path.join(cat_dir, f'receipt_{idx_str}_rot0.jpg'), quality=90)
    img.rotate(-90, expand=True).save(
        os.path.join(cat_dir, f'receipt_{idx_str}_rot90.jpg'), quality=90
    )
    img.rotate(180, expand=True).save(
        os.path.join(cat_dir, f'receipt_{idx_str}_rot180.jpg'), quality=90
    )
    img.rotate(90, expand=True).save(
        os.path.join(cat_dir, f'receipt_{idx_str}_rot270.jpg'), quality=90
    )


def parse_prediction(output_str):
  if not output_str or not output_str.strip():
    return -1

  s = output_str.strip()
  match_deg = re.search(r'\b(180°|270°|90°|0°)\b', s)
  if match_deg:
    return {'0°': 0, '90°': 1, '180°': 2, '270°': 3}[match_deg.group(1)]

  if re.search(r'upside|180\s*deg', s, re.I):
    return 2
  if re.search(r'counter|270\s*deg', s, re.I):
    return 3
  if re.search(r'clockwise|90\s*deg', s, re.I):
    return 1
  if re.search(r'upright|0\s*deg', s, re.I):
    return 0

  match_cls = re.search(
      r'(?:class|label|pred|result)[^\n]*?\b([0-3])\b', s, re.IGNORECASE
  )
  return int(match_cls.group(1)) if match_cls else -1


def run_evaluation(repo_dir, weights_path, test_local_dir):
  repo_dir = os.path.expanduser(repo_dir)
  predict_script = os.path.join(repo_dir, 'predict.py')
  abs_weights_path = (
      weights_path
      if os.path.isabs(weights_path)
      else os.path.join(repo_dir, weights_path)
  )

  if not os.path.exists(predict_script) or not os.path.exists(abs_weights_path):
    return

  python_bin = os.path.join(repo_dir, 'venv', 'bin', 'python')
  if not os.path.exists(python_bin):
    python_bin = (
        os.path.join(repo_dir, '.venv', 'bin', 'python')
        if os.path.exists(os.path.join(repo_dir, '.venv', 'bin', 'python'))
        else sys.executable
    )

  cat_dir = os.path.join(test_local_dir, 'receipt')
  if not os.path.exists(cat_dir):
    return

  files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.jpg')])
  total_count = len(files)
  if total_count == 0:
    return

  correct_count = 0
  for idx, fname in enumerate(files, 1):
    gt_label = (
        0
        if 'rot0.jpg' in fname
        else (
            1
            if 'rot90.jpg' in fname
            else 2 if 'rot180.jpg' in fname else 3 if 'rot270.jpg' in fname else -1
        )
    )
    if gt_label == -1:
      continue

    img_path = os.path.abspath(os.path.join(cat_dir, fname))
    cmd = [
        python_bin,
        'predict.py',
        '--image',
        img_path,
        '--weights',
        abs_weights_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_dir)
    pred_label = parse_prediction(result.stdout)

    if pred_label == gt_label:
      correct_count += 1

  acc = (correct_count / total_count) * 100
  print(f'\n[RECEIPT] - ACC: {acc:.2f}%')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--repo-dir',
      type=str,
      default=os.path.expanduser('~/target/a4-document-orientation'),
  )
  parser.add_argument(
      '--weights', type=str, default='weights/best_multiscale_fusion.pth'
  )
  args = parser.parse_args()

  current_dir = os.path.dirname(os.path.abspath(__file__))
  prepare_real_kaggle_dataset(current_dir, num_base=100)
  run_evaluation(args.repo_dir, args.weights, current_dir)
