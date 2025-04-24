# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal embeddings.
    takes 1d tensor of timesteps and embedding dimension
    returns tensor shape [len(timesteps), embedding_dim] w/ sinusoidal embeddings.
    """
    half_dim = embedding_dim // 2
    # Compute the frequency scales
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class ResidualBlock(nn.Module):
    """
    A residual block that incorporates a time embedding.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(16, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(16, out_channels)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        
        # Use a 1x1 conv if the number of channels change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        
        # Process and add the time embedding
        t_emb_out = self.time_emb(t_emb)
        t_emb_out = t_emb_out.unsqueeze(-1).unsqueeze(-1)  # reshape for broadcasting
        h = h + t_emb_out
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        return h + self.shortcut(x)

class Downsample(nn.Module):
    """
    Downsamples the spatial resolution by a factor of 2.
    """
    def __init__(self, channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """
    Upsamples the spatial resolution by a factor of 2.
    """
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    """
    Takes noisy image and a timestep as input and returns a prediction of the noise component present in the image.
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=128, time_emb_dim=256):
        super(UNet, self).__init__()
        self.time_emb_dim = time_emb_dim
        
        # converts the sinusoidal embedding into a learned embedding.
        self.time_embedding = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # init conv layer
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # downsampling path
        self.res1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = Downsample(base_channels)
        
        self.res2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = Downsample(base_channels * 2)

        # bottleneck
        self.res_mid = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        # upsampling path
        self.up1 = Upsample(base_channels * 2)
        self.res3 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_emb_dim)
        
        self.up2 = Upsample(base_channels)
        self.res4 = ResidualBlock(base_channels + base_channels, base_channels, time_emb_dim)
        
        # final output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        """
        Forward pass of the U-Net.
        x (torch.Tensor): Input noisy image of shape (B, C, H, W).
        t (torch.Tensor): Timestep tensor of shape (B,).
        """
        # Create and process the time embedding
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_embedding(t_emb)
        
        # Encoder
        x0 = self.init_conv(x)
        h1 = self.res1(x0, t_emb)
        h1_down = self.down1(h1)
        
        h2 = self.res2(h1_down, t_emb)
        h2_down = self.down2(h2)
        
        # Bottleneck
        h_mid = self.res_mid(h2_down, t_emb)
        
        # Decoder with skip connections
        up1 = self.up1(h_mid)
        cat1 = torch.cat([up1, h2], dim=1)
        h3 = self.res3(cat1, t_emb)
        
        up2 = self.up2(h3)
        cat2 = torch.cat([up2, h1], dim=1)
        h4 = self.res4(cat2, t_emb)
        
        # Output layer: returns the noise prediction
        out = self.out_conv(h4)
        return out

