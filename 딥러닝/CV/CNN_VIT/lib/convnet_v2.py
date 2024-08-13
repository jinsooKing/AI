import torch.nn as nn

from timm.models.layers import DropPath


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, stride=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, 0),
            nn.BatchNorm2d(out_dim),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, kernel_size=3, expand_ratio=2, droppath=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim * expand_ratio),
            nn.GELU(),
            nn.Conv2d(dim*expand_ratio, dim*expand_ratio, kernel_size = kernel_size, stride=1,
                      padding = (kernel_size-1)//2, groups=dim*expand_ratio, bias=False),
            nn.BatchNorm2d(dim*expand_ratio),
            nn.GELU(),
            nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.droppath = DropPath(droppath)

    def forward(self, x):
        x = x + self.droppath((self.layers(x)))
        return x

#################
class ConvNet(nn.Module):
    def __init__(self, blocks, dims, droppath, dropout, num_classes):
        super().__init__()
        self.downsamples = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.downsamples.append(Downsample(3, dims[0], 3, 1))
            else:
                self.downsamples.append(Downsample(dims[i-1], dims[i], 2, 2))

        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(nn.Sequential(*[Block(dims[i], kernel_size = 3, expand_ratio = 2, droppath = droppath) for _ in range(blocks[i])]))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
    
    def forward(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        x = self.dropout(self.norm(x.mean([-1, -2])))
        x = self.head(x)
        return x

####################