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


class Block(nn.Module):
    def __init__(self, dim, heads=8, expansion_ratio=2, droppath=0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mixer = MSA(dim, heads)
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
        self.in_chans = 3
        self.patch_size = patch_size
        self.patch_embed = nn.Linear((patch_size ** 2) * in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty((1, num_patches, embed_dim)))

    def forward(self, x):
        B, C, H, W = x.shape # B : batch size, C : Channel = 3 , H : height, W : width
        print(x.size())
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # 세로로 자르고 가로로 자르고 -> patchify
        print(x.size())
        x = x.reshape(B, C, -1, self.patch_size * self.patch_size).transpose(2, 3)
        print(x.size())

        patch_flat_size = self.patch_size * self.patch_size * self.in_chans
        print(x.size())
        x = x.reshape(B, -1, patch_flat_size)
        print(x.size())
        x = self.patch_embed(x)
        x = x + self.pos_embed
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self, blocks=12, embed_dim=384, num_classes=10, image_size=32, 
            patch_size=4, num_heads=8, expansion_ratio=2, droppath=0.0, dropout=0.0
        ):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(3, embed_dim, patch_size, num_patches)
        self.layers = nn.Sequential(*[
            Block(
                embed_dim, 
                num_heads, 
                expansion_ratio,
                droppath,
            )  
            for _ in range(blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.layers(x)
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
