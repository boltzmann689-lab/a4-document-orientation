import os
import subprocess

DATASETS = [
    'urbikn/sroie-datasetv2',
    'lonelvino/cord-1000',
    'dhiaznaidi/receiptdatasetssd300v2',
    'pdavpoojan/the-rvlcdip-dataset-test',
]

DOWNLOAD_DIR = os.path.join('data', 'raw')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print('Downloading raw datasets from Kaggle...')

for ds in DATASETS:
  folder_name = ds.split('/')[-1]
  target_path = os.path.join(DOWNLOAD_DIR, folder_name)
  os.makedirs(target_path, exist_ok=True)

  print(f'Fetching: {ds} ...')
  cmd = f"kaggle datasets download -d {ds} -p '{target_path}' --unzip"
  result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

  if result.returncode != 0:
    try:
      import kagglehub

      path = kagglehub.dataset_download(ds)
      print(f'Successfully downloaded via kagglehub to: {path}')
    except Exception as e:
      print(f'Failed to download {ds}: {e}')

print('Data acquisition complete.')
