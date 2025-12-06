import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

class DepthDataset(Dataset):
    def __init__(self, manifest_csv, img_size=(256, 256), dataset_root=None, exclude_phantom_types=None):
        self.df = pd.read_csv(manifest_csv)
        self.img_size = img_size

        # Filter out specified phantom types
        if exclude_phantom_types:
            self.df = self.df[~self.df['phantom_type'].isin(exclude_phantom_types)]
            self.df = self.df.reset_index(drop=True)

        # If dataset_root not provided, infer from manifest_csv location
        if dataset_root is None:
            # manifest_csv is in configs/, output root is its parent's parent
            manifest_path = Path(manifest_csv)
            # configs/dataset_manifest.csv -> output/
            self.dataset_root = manifest_path.parent.parent / "output"
        else:
            self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full paths
        img_path = self.dataset_root / row['image_path']
        depth_path = self.dataset_root / row['depth_path']

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        # Load depth
        depth = np.load(str(depth_path))
        depth = cv2.resize(depth, self.img_size)
        depth = torch.tensor(depth).unsqueeze(0).float()

        return img, depth