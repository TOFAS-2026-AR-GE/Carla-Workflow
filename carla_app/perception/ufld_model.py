"""UFLD ResNet-18 ağ tanımı.

Mimari, MIT lisanslı Ultra-Fast-Lane-Detection uygulamasıyla ve modelin
yayınlandığı CARLA deposundaki ``parsingNet`` durum sözlüğüyle uyumludur.
"""


def build_ufld_resnet18(grid_size=100, row_count=56, lane_count=4):
    """Yalnız model etkinleştirildiğinde PyTorch/torchvision yükler."""
    import torch
    from torch import nn
    from torchvision.models import resnet18

    class ResNetBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = resnet18(weights=None)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

        def forward(self, tensor):
            tensor = self.conv1(tensor)
            tensor = self.bn1(tensor)
            tensor = self.relu(tensor)
            tensor = self.maxpool(tensor)
            tensor = self.layer1(tensor)
            feature2 = self.layer2(tensor)
            feature3 = self.layer3(feature2)
            feature4 = self.layer4(feature3)
            return feature2, feature3, feature4

    class ParsingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_dim = (grid_size + 1, row_count, lane_count)
            self.model = ResNetBackbone()
            self.pool = nn.Conv2d(512, 8, 1)
            self.cls = nn.Sequential(
                nn.Linear(1800, 2048),
                nn.ReLU(),
                nn.Linear(2048, int(torch.tensor(self.cls_dim).prod().item())),
            )

        def forward(self, tensor):
            _feature2, _feature3, feature4 = self.model(tensor)
            pooled = self.pool(feature4).reshape(-1, 1800)
            return self.cls(pooled).reshape(-1, *self.cls_dim)

    return ParsingNet()
