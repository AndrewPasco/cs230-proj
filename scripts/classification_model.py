import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Freeze all but last 2 layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze layers for transfer learning
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # Binary classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(  # type: ignore
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class MobileNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()

        self.backbone = models.mobilenet_v3_small(
            weights="IMAGENET1K_V1" if pretrained else None
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze the last few blocks of the feature extractor for transfer learning
            total_layers = len(self.backbone.features)
            for i in range(total_layers - 4, total_layers):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = True

        # Extract the input features from that last layer
        last_channel = self.backbone.classifier[0].in_features

        # Replace the classifier head with binary classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(last_channel, 256),
            nn.Hardswish(),  # standard in MobileNet, faster than ReLU
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Ensure head is trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
