import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from classification_model import ResNetBinaryClassifier
from dataset import CVATDataset
import train_utils


# --- Setup ---
train_utils.setup_single_threaded_torch()
DEVICE = train_utils.get_device()

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATASET_DIR = os.path.join(ROOT_DIR, "data")
IMG_SIZE = (240, 320)
SAVE_NAME = "classifier_resnet18.pth.tar"

EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4
VAL_SPLIT = 0.2


# --- Training & Validation Loops ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    batch_idx = 1
    total_batches = len(loader)
    for batch in loader:
        imgs = batch["input"].to(device)
        labels = batch["has_feature"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (batch_idx) % 3 == 0 or (batch_idx) == total_batches:
            print(f"  Batch {batch_idx}/{total_batches} - loss: {loss.item():.4f}")

        batch_idx += 1

        running_loss += loss.item()

        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["input"].to(device)
            labels = batch["has_feature"].float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return val_loss / len(loader), correct / total


def main():
    # Load arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file to initialize the model with. Training will resume from the epoch saved in the checkpoint.",
    )
    args = parser.parse_args()

    # --- Dataset ---
    full_dataset = CVATDataset(
        DATASET_DIR, has_gt=False, img_size=IMG_SIZE, for_classification=True
    )

    # Train/val split
    random_generator = torch.Generator().manual_seed(42)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=random_generator
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model, Loss, Optimizer ---
    model = ResNetBinaryClassifier(pretrained=True, freeze_backbone=True).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Load checkpoint if provided
    if args.checkpoint:
        model, epoch, metric = train_utils.load_checkpoint(
            model, args.checkpoint, DEVICE
        )
        best_acc = metric["best_acc"]
        train_loss_list = metric["train_loss_list"]
        val_loss_list = metric["val_loss_list"]
        train_acc_list = metric["train_acc_list"]
        val_acc_list = metric["val_acc_list"]
        print(
            f"Loaded a checkpoint from {args.checkpoint} at epoch {epoch} with best acc {best_acc}. Resuming training from epoch {epoch + 1}."
        )
        epoch += 1  # start training from the next epoch
    else:
        epoch = 1
        best_acc = float("-inf")
        train_loss_list, val_loss_list, train_acc_list, val_acc_list = (
            list(),
            list(),
            list(),
            list(),
        )
        print(
            f"No checkpoint provided. Starting training from scratch at epoch {epoch}."
        )

    while epoch <= EPOCHS:
        print(f"\nEpoch ({epoch}/{EPOCHS})")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            train_utils.save_checkpoint(
                model,
                epoch,
                {
                    "train_loss_list": train_loss_list,
                    "train_acc_list": train_acc_list,
                    "val_loss_list": val_loss_list,
                    "val_acc_list": val_acc_list,
                    "best_acc": best_acc,
                },
                filename=SAVE_NAME,
            )

        # Update learning curve plot
        train_utils.save_learning_curve(
            {
                "train_loss": train_loss_list,
                "val_loss": val_loss_list,
                "train_acc": train_acc_list,
                "val_acc": val_acc_list,
            },
            filename="classification_curve.png",
            title="Classification Training Progress",
        )

        epoch += 1

    print(f"Training complete. Best Val Acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()
