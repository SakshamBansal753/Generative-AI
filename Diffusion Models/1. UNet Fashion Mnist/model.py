import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_time_embedding(timesteps,dim):
    half_dim = dim // 2
    freqs=torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    )
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # zero pad
        embedding = F.pad(embedding, (0, 1,))
    return embedding
class TimeMLP(nn.Module):
    def __init__(self,dim,out_dim):
        super().__init__()
        self.dim=dim
        self.mlp=nn.Sequential(
            nn.Linear(dim,out_dim),
            nn.SiLU(),
            nn.Linear(out_dim,out_dim)
        )
    def forward(self,t):
        k=sinusoidal_time_embedding(t,self.dim)
        return self.mlp(k)
class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,time_dim,dropout=0.1):
        super().__init__()
        self.norm1=nn.GroupNorm(8,in_ch)
        self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.time_proj=nn.Linear(time_dim,out_ch)
        self.norm2=nn.GroupNorm(8,out_ch)
        self.dropout=nn.Dropout(dropout)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.skip=nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x,t_emb):
        h=self.conv1(F.silu(self.norm1(x)))
        h=h+self.time_proj(F.silu(t_emb))[:,:,None,None]
        h=self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h+self.skip(x)
class SelfAttention2d(nn.Module):
    """Simple single-head self-attention over spatial positions, used at low resolution."""
 
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
 
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 3, 2)  # (3,B,HW,C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(C), dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
 
    def forward(self, x):
        return self.op(x)
 
 
class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)
 
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)
 

class UNet(nn.Module):
    def __init__(self,in_ch=1,base_ch=64,time_dim=256,num_classes=None):
        super().__init__()
        self.time_mlp=TimeMLP(base_ch,time_dim)
        self.num_classes=num_classes
        if num_classes is not None:
            self.class_emb=nn.Embedding(num_classes+1,time_dim)
        self.in_conv=nn.Conv2d(in_ch,base_ch,3,padding=1)
        self.down_res1=ResidualBlock(base_ch,base_ch,time_dim)
        self.pool1=Downsample(base_ch)
        self.down_res2=ResidualBlock(base_ch,base_ch*2,time_dim)
        self.pool2=Downsample(base_ch*2)

        self.boot1=ResidualBlock(base_ch*2,base_ch*4,time_dim)
        self.boot_attn=SelfAttention2d(base_ch*4)
        self.bott2=ResidualBlock(base_ch*4,base_ch*2,time_dim)

        self.up1=Upsample(base_ch*2)
        self.up_res1=ResidualBlock(base_ch*2+base_ch*2,base_ch*2,time_dim)
        self.up2=Upsample(base_ch*2)
        self.up1_res2=ResidualBlock(base_ch*2+base_ch,base_ch,time_dim)

        self.out_norm=nn.GroupNorm(8,base_ch)
        self.out_conv=nn.Conv2d(base_ch,in_ch,3,padding=1)
    def forward(self,x,t,y=None):
        t_emb=self.time_mlp(t)
        if self.num_classes is not None:
            assert y is not None
            t_emb=t_emb+self.class_emb(y)
        h0=self.in_conv(x)
        skip_a=self.down_res1(h0,t_emb)
        h=self.pool1(skip_a)
        skip_b=self.down_res2(h,t_emb)
        h=self.pool2(skip_b)

        h=self.boot1(h,t_emb)
        h=self.boot_attn(h)
        h=self.bott2(h,t_emb)

        h=self.up1(h)
        h=self.up_res1(torch.cat([h,skip_b],dim=1),t_emb)
        
        h=self.up2(h)
        h=self.up1_res2(torch.cat([h,skip_a],dim=1),t_emb)

        

        out=self.out_conv(F.silu(self.out_norm(h)))
        return out
