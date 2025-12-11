import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class UT_HAR_MLP(nn.Module):
    def __init__(self):
        super(UT_HAR_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(250*90,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,7)
        )
        
    def forward(self,x):
        x = x.view(-1,250*90)
        x = self.fc(x)
        return x


class UT_HAR_RNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.rnn = nn.RNN(90, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        x = x.view(-1,250,90).permute(1,0,2)
        _, ht = self.rnn(x)
        return self.fc(ht[-1])


# --------------------------
# Vision Transformer (ViT)
# --------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size_w=50, patch_size_h=18,
                 emb_size=900, img_size=250*90):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=(patch_size_w, patch_size_h),
                      stride=(patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        num_patches = img_size // emb_size
        self.position = nn.Parameter(torch.randn(num_patches + 1, emb_size))

    def forward(self, x):
        b = x.size(0)
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=900, num_heads=5, dropout=0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x),
                        "b n (h d qkv) -> qkv b h n d",
                        h=self.num_heads, qkv=3)
        queries, keys, values = qkv

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            energy = energy.masked_fill(~mask, float('-inf'))

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=900, drop_p=0.0,
                 forward_expansion=4, forward_drop_p=0.0, **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size,
                                 expansion=forward_expansion,
                                 drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=1, **kwargs):
        super().__init__(*[
            TransformerEncoderBlock(**kwargs) for _ in range(depth)
        ])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=900, n_classes=7):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


class UT_HAR_ViT(nn.Sequential):
    def __init__(self, in_channels=1,
                 patch_size_w=50, patch_size_h=18,
                 emb_size=900, img_size=250*90,
                 depth=1, n_classes=7, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
