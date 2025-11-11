import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils
import cv2


class CVATDataset(Dataset):
    def __init__(
        self, dataset_dir, has_gt=True, img_size=(240, 320), for_classification=False
    ):
        """
        Args:
            dataset_dir: root folder containing 'images' and 'masks'
            has_gt: True if ground truth masks are available
            img_size: (H, W) resize for network input
            for_classification: if True, return binary label 'has_feature' instead of segmentation mask
        """
        self.dataset_dir = dataset_dir
        self.has_gt = has_gt
        self.img_size = img_size  # (H, W)
        self.for_classification = for_classification

        # Transform for RGB images
        mean_rgb = [0.499, 0.493, 0.598]
        std_rgb = [0.217, 0.212, 0.177]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_rgb, std=std_rgb),
            ]
        )

        # Collect image and mask names
        images_path = os.path.join(dataset_dir, "images")
        masks_path = os.path.join(dataset_dir, "masks")

        self.img_files = sorted(os.listdir(images_path))
        self.mask_basenames = (
            {os.path.splitext(f)[0] for f in os.listdir(masks_path)}
            if os.path.exists(masks_path)
            else set()
        )

        # Precompute feature labels (Assumes mask presence indicates feature presence)
        self.has_feature_list = [
            1 if os.path.splitext(f)[0] in self.mask_basenames else 0
            for f in self.img_files
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = os.path.splitext(self.img_files[idx])[0]
        img_path = os.path.join(self.dataset_dir, "images", self.img_files[idx])

        # Load and preprocess image
        img = utils.read_rgb(img_path)
        img = cv2.resize(
            img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR
        )
        img = self.transform(img)

        # Basic fields
        sample = {
            "input": img,
            "filename": filename,
            "has_feature": torch.tensor(
                self.has_feature_list[idx], dtype=torch.float32
            ),
        }

        # For classification tasks
        if self.for_classification:
            sample["target"] = sample["has_feature"]

        # For segmentation tasks (if GT available)
        elif self.has_gt and sample["has_feature"] == 1:
            mask_path = os.path.join(self.dataset_dir, "masks", f"{filename}.png")
            mask = utils.read_mask(mask_path)
            mask = cv2.resize(
                mask,
                (self.img_size[1], self.img_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["target"] = torch.LongTensor(mask)

        return sample
