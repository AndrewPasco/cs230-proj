# classification_model.py
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Optionally freeze all but last 2 layers
        if freeze_backbone:
            # Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze layers
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # Replace FC layer with custom binary classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(  # type: ignore
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Ensure new head is trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
