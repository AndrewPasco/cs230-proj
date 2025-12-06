import torch
from torch.utils.data import DataLoader, random_split
from dataset import DepthDataset
from model import get_model
import sys
from pathlib import Path
import csv
from train_utils import save_learning_curve
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Setup
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...", flush=True)
model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
print("Model loaded", flush=True)

# Checkpoint loading
checkpoint_dir = Path(__file__).parent / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
latest_checkpoint = sorted(checkpoint_dir.glob("epoch_*.pth"))[-1] if list(checkpoint_dir.glob("epoch_*.pth")) else None
start_epoch = 0

# CSV logging setup
csv_path = checkpoint_dir / "training_log.csv"
csv_exists = csv_path.exists()

if latest_checkpoint:
    print(f"Loading checkpoint: {latest_checkpoint}", flush=True)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}", flush=True)

# Initialize CSV if starting fresh
if start_epoch == 0 and not csv_exists:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_ssim'])

# Data
print("Loading dataset...", flush=True)
manifest_csv = "C:/Users/georg/Documents/Stanford/cs230/cs230-proj/scripts/labeling/output/dataset_manifest.csv"
full_dataset = DepthDataset(manifest_csv, exclude_phantom_types=['p4'])

# Split into train/val
val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                           generator=torch.Generator().manual_seed(42))

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"DataLoader ready: {len(train_loader)} batches", flush=True)

train_losses = []
val_losses = []

# Train
print("\nStarting training...\n", flush=True)
for epoch in range(start_epoch, 50):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, (imgs, depths) in enumerate(train_loader):
        imgs, depths = imgs.to(device), depths.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, depths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # Print every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} - Batch {batch_idx+1}/{len(train_loader)}: {loss.item():.6f}", flush=True)

    train_loss = total_loss / batch_count    
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{50}: Avg Train Loss = {train_loss:.6f}", flush=True)
    
    # Validation
    model.eval()
    val_loss = 0
    val_mae = 0
    val_ssim = 0
    with torch.no_grad():
        for imgs, depths in val_loader:
            imgs, depths = imgs.to(device), depths.to(device)
            preds = model(imgs)
            val_loss += loss_fn(preds, depths).item()
            val_mae += F.l1_loss(preds, depths).item()
            val_ssim += ssim_metric(preds, depths).item()
    val_loss /= len(val_loader)
    val_mae /= len(val_loader)
    val_ssim /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{50}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}", flush=True)

    # Log to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, val_loss, val_mae, val_ssim])

    # Save checkpoint every epoch (safe save: write to temp file first)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    temp_path = checkpoint_dir / f"epoch_{epoch:03d}.pth.tmp"
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': train_loss
    }, temp_path)
    # Atomic rename (safe)
    temp_path.rename(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}", flush=True)

print("\nTraining complete!", flush=True)

save_learning_curve(
    metrics={"train_loss": train_losses, "val_loss": val_losses},
    filename="depth_learning_curve.png"
)

# Save final model
print("Saving final model...", flush=True)
torch.save(model.state_dict(), "depth_model_final.pth")
print("Model saved to depth_model_final.pth", flush=True)