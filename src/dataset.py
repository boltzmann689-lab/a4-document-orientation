import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocess import simulate_kaggle_capture_pipeline


class DocumentDataset(Dataset):
    """Single-scale Document Dataset for the baseline (not used in the fusion pipeline)."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class MultiScaleDocumentDataset(Dataset):
    """
    Multi-Scale Document Dataset - matches the Kaggle patch extraction logic
    exactly (PIL -> numpy -> slicing -> cv2.resize). DO NOT modify this
    algorithm, as the model was trained using this exact patch extraction method.

    New parameter compared with the original Git version:
    -------------------------------------------------------
    simulate_capture_pipeline (bool):
        - True  -> Apply simulate_kaggle_capture_pipeline() to the image BEFORE
                   patch extraction/transformation. USED FOR RAW images on the
                   local machine (images that have not gone through the
                   resize-640 + JPEG-85 processing step of the Kaggle dataset
                   generation script). => This flag will be enabled by predict.py.
        - False (default) -> Used when loading the existing image files from
                   real_text_10k_dataset/ (which have already been processed
                   with resize + compression). Enabling True in this case would
                   result in double JPEG compression, shifting the distribution
                   in a different direction => keep it disabled.
    """

    def __init__(
        self,
        df,
        transform_global=None,
        transform_patch=None,
        simulate_capture_pipeline: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform_global = transform_global
        self.transform_patch = transform_patch
        self.simulate_capture_pipeline = simulate_capture_pipeline

    def __len__(self):
        return len(self.df)

    def extract_center_patches(self, img_pil):
        """Extract 4 patches from the central 2x2 grid - identical to the Kaggle version."""
        img_np = np.array(img_pil)
        h, w, _ = img_np.shape
        h_step, w_step = h // 3, w // 3

        p1 = img_np[h_step // 2 : h_step // 2 + h_step, w_step // 2 : w_step // 2 + w_step]
        p2 = img_np[
            h_step // 2 : h_step // 2 + h_step,
            w_step + w_step // 2 : w_step + w_step // 2 + w_step,
        ]
        p3 = img_np[
            h_step + h_step // 2 : h_step + h_step // 2 + h_step,
            w_step // 2 : w_step // 2 + w_step,
        ]
        p4 = img_np[
            h_step + h_step // 2 : h_step + h_step // 2 + h_step,
            w_step + w_step // 2 : w_step + w_step // 2 + w_step,
        ]

        patches = [p1, p2, p3, p4]
        patch_pils = []
        for p in patches:
            if p.size == 0 or p.shape[0] == 0 or p.shape[1] == 0:
                p = cv2.resize(img_np, (112, 112))
            else:
                p = cv2.resize(p, (112, 112))
            patch_pils.append(Image.fromarray(p))
        return patch_pils

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label"]

        image = Image.open(img_path).convert("RGB")

        if self.simulate_capture_pipeline:
            image = simulate_kaggle_capture_pipeline(image)

        global_img = self.transform_global(image) if self.transform_global else image
        patches = self.extract_center_patches(image)

        if self.transform_patch:
            patches_tensor = torch.stack([self.transform_patch(p) for p in patches])
        else:
            patches_tensor = patches

        return global_img, patches_tensor, torch.tensor(label, dtype=torch.long)