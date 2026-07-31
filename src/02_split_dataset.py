import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Scan directory and extract metadata
base_dir = "data_1000"
data_list = []

for sub in os.listdir(base_dir):
    sub_path = os.path.join(base_dir, sub)
    if os.path.isdir(sub_path):
        for img_name in os.listdir(sub_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(sub_path, img_name)
                data_list.append({
                    "image_path": full_path,
                    "sub_category": sub
                })

df = pd.DataFrame(data_list)

# Stratified Split: 70% Train, 15% Val, 15% Test
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    stratify=df['sub_category'], 
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    stratify=temp_df['sub_category'], 
    random_state=42
)

# Export split metadata to CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Dataset Split Summary
report = pd.DataFrame({
    'Train (70%)': train_df['sub_category'].value_counts(),
    'Val (15%)': val_df['sub_category'].value_counts(),
    'Test (15%)': test_df['sub_category'].value_counts(),
    'Total': df['sub_category'].value_counts()
})

print("\n" + "="*50)
print("STRATIFIED DATASET SPLIT REPORT")
print("="*50)
print(report)
print("-" * 50)
print(f"Total: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | All={len(df)}")
print("="*50)