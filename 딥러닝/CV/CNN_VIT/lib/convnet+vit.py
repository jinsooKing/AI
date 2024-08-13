import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class MSA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv = [qkv_part.reshape(qkv_part.shape[0], -1, self.heads, qkv_part.shape[-1] // self.heads) for qkv_part in qkv]
        q, k, v =  [qkv_part.transpose(1, 2).clone() for qkv_part in qkv]  

        q *= self.scale
        k *= self.scale

        att_score = torch.matmul(q, k.transpose(-2, -1))
        att_weights = F.softmax(att_score, dim=-1)
        att_output = torch.matmul(att_weights, v)
        out = att_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.heads * (x.size(-1) // self.heads))
        x = self.to_out(out)

        return x

class MLP(nn.Module):
    def __init__(self, dim, expansion_ratio=2):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x    


class tBlock(nn.Module):
    def __init__(self, dim, heads=4, expansion_ratio=2, droppath=0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mixer = MSA(dim=dim, heads=heads)
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mixer = MLP(dim, expansion_ratio)
        self.droppath = DropPath(droppath)
    
    def forward(self, x):
        x = x + self.droppath(self.token_mixer(self.token_norm(x)))
        x = x + self.droppath(self.channel_mixer(self.channel_norm(x)))

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, num_patches):
        super().__init__()
        self.in_chans = 128
        self.patch_size = patch_size
        self.patch_embed = nn.Linear((patch_size ** 2) * in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty((1, num_patches, embed_dim)))
        
    def forward(self, x):
        B, C, H, W = x.shape  # B: batch size, C: Channel = 3 , H: height, W: width
        unfold_h = (H // self.patch_size) * self.patch_size
        unfold_w = (W // self.patch_size) * self.patch_size
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)  # 세로로 자르고 가로로 자르고 -> patchify
        x = x[:, :, :unfold_h, :unfold_w, :, :]  # 이미지 크기를 패치 크기의 배수로 맞춤
        x = x.reshape(B, C, -1, self.patch_size * self.patch_size).transpose(2, 3)
        patch_flat_size = self.patch_size * self.patch_size * self.in_chans
        x = x.reshape(B, -1, patch_flat_size)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        return x




class VisionTransformer(nn.Module):
    def __init__(
            self, vit_blocks=12, vit_embed_dim=384, num_classes=10, image_size=15, 
            patch_size=3, vit_num_heads=4, vit_expansion_ratio=2, vit_droppath=0.0, vit_dropout=0.0
        ):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(128, vit_embed_dim, patch_size, num_patches)
        self.tlayers = nn.Sequential(*[
            tBlock(
                vit_embed_dim, 
                vit_num_heads, 
                vit_expansion_ratio,
                vit_droppath,
            )  
            for _ in range(vit_blocks)]
        )
        self.dropout = nn.Dropout(vit_dropout)
        self.norm = nn.LayerNorm(vit_embed_dim)
        self.head = nn.Linear(vit_embed_dim, num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.tlayers(x)
        x = self.dropout(self.norm(x.mean([1])))
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (PatchEmbedding)):
            trunc_normal_(m.pos_embed, std=0.02)

############################################################################### up : vit / down : convnet_v2

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





class ConvNet(nn.Module):
    def __init__(self, blocks, dims, droppath):
        super().__init__()
        self.downsamples = nn.ModuleList()
        for i in range(2):
            if i == 0:
                self.downsamples.append(Downsample(3, dims[0], 3, 1))
            else:
                self.downsamples.append(Downsample(dims[i-1], dims[i], 2, 2))

        self.layers = nn.ModuleList()
        for i in range(2):
            self.layers.append(nn.Sequential(*[Block(dims[i], kernel_size = 3, expand_ratio = 2, droppath = droppath) for _ in range(blocks[i])]))
    
    def forward(self, x):
        for i in range(2):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
            
        return x

######################################################################## down : combined model

class CONVIT(nn.Module):
    def __init__(self,
                 blocks, dims, droppath,
                 vit_blocks=12, vit_embed_dim=384, num_classes=10, image_size=15, 
                 patch_size=3, vit_num_heads=4, vit_expansion_ratio=2, vit_droppath=0.0, vit_dropout=0.0
                ):
        super().__init__()
        self.convnet = ConvNet(blocks, dims, droppath)
        self.vit = VisionTransformer(vit_blocks=12, vit_embed_dim=384, num_classes=10, image_size=15, 
                                    patch_size=3, vit_num_heads=4, vit_expansion_ratio=2, vit_droppath=0.0, vit_dropout=0.0)

    def forward(self, x):
        
        x = self.convnet(x)
        output = self.vit(x)

        return output















