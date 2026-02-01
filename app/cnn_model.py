import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c, c // r)
        self.fc2 = nn.Linear(c // r, c)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep

class BottleneckBlock(nn.Module):
    def __init__(self, c, bottleneck_ratio=4, drop_path=0.0):
        super().__init__()
        mid = c // bottleneck_ratio

        self.conv1 = nn.Conv2d(c, mid, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(mid, c, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(c)

        self.se = SEBlock(c, r=8)
        self.dp = DropPath(drop_path)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.dp(out)
        return F.relu(x + out, inplace=True)

class Connect4StrongCNN(nn.Module):
    """
    Input:  (B,2,6,7)
    Output: (B,7)
    """
    def __init__(self, channels=192, n_blocks=12, drop_path_max=0.10, head_dropout=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        dps = torch.linspace(0, drop_path_max, n_blocks).tolist()
        self.blocks = nn.Sequential(*[
            BottleneckBlock(channels, bottleneck_ratio=4, drop_path=dps[i])
            for i in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
