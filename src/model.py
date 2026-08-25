"""Small CNN baseline for 9-class wafer defect classification."""

from torch import nn


class WaferCNN(nn.Module):
    """A compact 3-block CNN.

    Input is a 2-channel wafer map: channel 0 is a "die exists" mask, channel 1
    is a "die failed" mask (see src.dataset.WaferMapDataset). Two binary channels
    are used rather than a single ordinal {0,.5,1} channel so the network isn't
    told that a passing die sits "halfway between" background and a failed die --
    background / pass / fail are categories, not points on a scale.
    """

    def __init__(self, num_classes: int = 9, in_size: int = 64, in_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced = in_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * reduced * reduced, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
