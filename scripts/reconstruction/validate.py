"""
Validate depth prediction model on test images.

Compares predicted depth vs ground truth depth and visualizes results.
"""

import torch
from torch.utils.data import DataLoader
from dataset import DepthDataset
from model import get_model
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")
    return model

def validate(manifest_csv, checkpoint_path=None, num_samples=10, img_size=(256, 256),
             test_phantom_types=None, test_run_ids=None, checkpoint_dir_name="checkpoints"):
    """Validate model on test set.

    Args:
        manifest_csv: Path to dataset_manifest.csv
        checkpoint_path: Path to checkpoint (auto-finds latest if None)
        num_samples: Number of samples to validate (None = all samples)
        img_size: Image size for model input
        test_phantom_types: List of phantom types to use for testing (e.g., ['p3'])
        test_run_ids: List of specific run IDs to test on (e.g., ['p1_n2t2n_25N-2025-11-25_23.05.56'])
        checkpoint_dir_name: Name of checkpoint directory to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model
    model = get_model().to(device)

    if checkpoint_path is None:
        # Auto-find latest checkpoint
        checkpoint_dir = Path(__file__).parent / checkpoint_dir_name
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"))
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}! Train the model first.")
            return
        checkpoint_path = checkpoints[-1]

    model = load_checkpoint(checkpoint_path, model, device)
    model.eval()

    # Load dataset
    print(f"\nLoading dataset: {manifest_csv}")
    dataset = DepthDataset(manifest_csv, img_size=img_size)

    # Filter test indices based on criteria
    test_indices = list(range(len(dataset)))

    if test_run_ids:
        print(f"Testing on specific runs: {test_run_ids}")
        test_indices = [i for i in test_indices
                       if dataset.df.iloc[i]['run_id'] in test_run_ids]
        print(f"Test set size: {len(test_indices)} samples (from {len(test_run_ids)} runs)\n")
    elif test_phantom_types:
        print(f"Testing on phantom types: {test_phantom_types}")
        test_indices = [i for i in test_indices
                       if dataset.df.iloc[i]['phantom_type'] in test_phantom_types]
        print(f"Test set size: {len(test_indices)} samples\n")
    else:
        print(f"Dataset size: {len(dataset)} samples\n")

    # Determine which samples to validate
    if num_samples is None or num_samples >= len(test_indices):
        # Use all test samples
        indices = test_indices
        print(f"Validating on ALL {len(indices)} test samples:\n")
    else:
        # Random subset
        indices = np.random.choice(test_indices, min(num_samples, len(test_indices)), replace=False)
        print(f"Validating on {len(indices)} random samples:\n")

    errors = []
    ssims = []

    for idx, sample_idx in enumerate(indices):
        img, gt_depth = dataset[sample_idx]
        phantom_type = dataset.df.iloc[sample_idx]['phantom_type']
        run_id = dataset.df.iloc[sample_idx]['run_id']

        # Add batch dimension
        img_batch = img.unsqueeze(0).to(device)
        gt_depth_batch = gt_depth.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred_depth = model(img_batch)

        # Compute error
        valid_mask = gt_depth_batch > 0
        if valid_mask.sum() > 0:
            error = torch.abs(pred_depth - gt_depth_batch)[valid_mask].mean().item()
            errors.append(error)

            # Compute SSIM
            from skimage.metrics import structural_similarity as ssim
            gt_np = gt_depth.cpu().numpy().squeeze()
            pred_np = pred_depth.cpu().detach().numpy().squeeze()
            valid_mask_np = (gt_np > 0)
            if valid_mask_np.sum() > 0:
                ssim_val = ssim(gt_np, pred_np, data_range=gt_np.max() - gt_np.min())
                ssims.append(ssim_val)

            print(f"Sample {idx+1}/{len(indices)} ({phantom_type} - {run_id}): MAE = {error:.6f} m, SSIM = {ssim_val:.4f}", flush=True)

        # Visualize
        if idx < 10:  # Show first 10 samples
            visualize_sample(img, gt_depth, pred_depth, sample_idx, error, phantom_type, run_id)

    # Summary
    if errors:
        print(f"\n{'='*50}")
        print(f"Validation Summary:")
        print(f"  Samples: {len(errors)}")
        print(f"  Mean Absolute Error: {np.mean(errors):.6f} m")
        print(f"  Std Dev: {np.std(errors):.6f} m")
        print(f"  Min: {np.min(errors):.6f} m")
        print(f"  Max: {np.max(errors):.6f} m")
        if ssims:
            print(f"  Mean SSIM: {np.mean(ssims):.4f}")
            print(f"  SSIM Std Dev: {np.std(ssims):.4f}")
        print(f"{'='*50}")


def visualize_sample(img, gt_depth, pred_depth, sample_idx, error, phantom_type, run_id):
    """Visualize a single sample."""
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    gt_np = gt_depth.cpu().numpy().squeeze()
    pred_np = pred_depth.cpu().detach().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Input image
    axes[0].imshow((img_np * 255).astype(np.uint8))
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Ground truth depth
    valid_gt = gt_np > 0
    axes[1].imshow(gt_np, cmap='jet')
    axes[1].set_title("Ground Truth Depth")
    axes[1].axis('off')
    cbar1 = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Depth (m)')

    # Predicted depth
    axes[2].imshow(pred_np, cmap='jet')
    axes[2].set_title(f"Predicted Depth\nMAE: {error:.6f} m")
    axes[2].axis('off')
    cbar2 = plt.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label('Depth (m)')

    plt.suptitle(f"{phantom_type} - {run_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"validation_{phantom_type}_{sample_idx}.png", dpi=100, bbox_inches='tight')
    print(f"  → Saved: validation_{phantom_type}_{sample_idx}.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate depth prediction model")
    parser.add_argument(
        "--manifest",
        type=str,
        default="C:/Users/georg/Documents/Stanford/cs230/cs230-proj/scripts/labeling/output/dataset_manifest.csv",
        help="Path to dataset_manifest.csv"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint (default: latest)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory name (default: checkpoints)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to validate (default: all)"
    )
    parser.add_argument(
        "--test-phantoms",
        type=str,
        nargs="+",
        help="Phantom types for test set (e.g., --test-phantoms p3 p4)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Depth Model Validation")
    print("="*60 + "\n")

    validate(
        args.manifest,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        test_phantom_types=args.test_phantoms,
        checkpoint_dir_name=args.checkpoint_dir
    )
