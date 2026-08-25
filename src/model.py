import torch
import torch.nn as nn
from torchvision import models


class MultiScaleResNet(nn.Module):
  """Multi-Scale Fusion Architecture fusing global layout features with 4 local patch features."""

  def __init__(self, num_classes=4):
    super(MultiScaleResNet, self).__init__()

    # Global Image Stream Backbone
    global_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    self.global_backbone = nn.Sequential(*list(global_resnet.children())[:-1])

    # Local Patch Stream Backbone
    patch_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    self.patch_backbone = nn.Sequential(*list(patch_resnet.children())[:-1])

    # Fusion Classifier (512 + 4 * 512 = 2560 dims -> 512 -> num_classes)
    self.classifier = nn.Sequential(
        nn.Linear(512 + 512 * 4, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )

  def forward(self, global_img, patches):
    # Extract Global Features
    g_feats = self.global_backbone(global_img).squeeze(-1).squeeze(-1)

    # Extract Local Patch Features
    bs, num_patches, c, h, w = patches.size()
    patches_flat = patches.view(bs * num_patches, c, h, w)
    p_feats = self.patch_backbone(patches_flat).view(bs, num_patches * 512)

    # Feature Fusion
    combined_feats = torch.cat([g_feats, p_feats], dim=1)

    # Output Logits
    out = self.classifier(combined_feats)
    return out
