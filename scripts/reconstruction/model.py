import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetWithHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="densenet161",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,   # still 1 output
        )

        # Deeper module Head
        self.unet.segmentation_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Freeze encoder 
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.unet(x)
    
def get_model():
    return UNetWithHead()