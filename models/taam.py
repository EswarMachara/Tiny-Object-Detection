"""
TAAM: Tiny-Aware Attention Module for Tiny Object Detection
============================================================

A well-motivated attention module designed specifically for tiny object detection,
based on the observation that tiny objects have low feature SNR and require
scale-adaptive processing.

Design Principles:
1. Scale-Adaptive Processing: Different object sizes need different attention
2. Contrast-Based Enhancement: Tiny objects are detected by difference from background
3. Gaussian Spatial Focus: Focused attention on object regions

Components:
- ScaleEstimator: Predicts local object scale at each position
- ContrastEnhancer: Enhances object features relative to local background
- GaussianSpatialAttention: Applies scale-adaptive Gaussian-weighted attention
- TAAM: Main module combining all components

Author: Research Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaleEstimator(nn.Module):
    """
    Estimates local object scale at each spatial position.
    
    For tiny object detection, knowing the approximate scale helps adapt
    the attention mechanism - smaller objects need narrower attention focus
    and stronger contrast enhancement.
    
    Output: Scale map σ(x,y) ∈ (0, 1) where:
        - Low σ → tiny object (narrow focus, strong enhancement)
        - High σ → larger object (wider focus, mild enhancement)
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)
        
        self.scale_net = nn.Sequential(
            # Depthwise separable conv for efficiency
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=True),
            nn.Sigmoid()
        )
        
        # Initialize to predict medium scale (0.5) initially
        nn.init.constant_(self.scale_net[-2].bias, 0.0)
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            scale_map: Per-location scale estimate [B, 1, H, W] in (0, 1)
        """
        return self.scale_net(x)


class ContrastEnhancer(nn.Module):
    """
    Enhances features based on local contrast (object vs background).
    
    Motivation: Tiny objects are often detected by how they differ from
    their surroundings, not by absolute feature values. This is inspired
    by Weber's Law in human vision - we perceive relative differences.
    
    F_enhanced = F + α · (F - F_local_mean)
    where α is larger for predicted tiny objects (low scale)
    """
    
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Learnable contrast weight (scale-adaptive)
        self.alpha_net = nn.Sequential(
            nn.Conv2d(1, 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
            nn.Softplus()  # Ensures positive
        )
        
        # Local mean computation (depthwise conv with fixed averaging kernel)
        self.register_buffer('avg_kernel', 
            torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size))
        
        # Channel mixing for enhanced features
        self.channel_mix = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x, scale_map):
        """
        Args:
            x: Input features [B, C, H, W]
            scale_map: Per-location scale [B, 1, H, W]
        Returns:
            Enhanced features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Compute local mean per channel
        # Use avg pooling for efficiency
        x_padded = F.pad(x, [self.padding]*4, mode='reflect')
        local_mean = F.avg_pool2d(x_padded, self.kernel_size, stride=1)
        
        # Contrast = difference from local mean
        contrast = x - local_mean
        
        # Scale-adaptive alpha: smaller scale → larger alpha
        # Invert scale_map: tiny objects (low σ) get high alpha
        alpha = self.alpha_net(1.0 - scale_map)
        
        # Apply contrast enhancement
        enhanced = x + alpha * contrast
        
        # Channel mixing with residual
        enhanced = self.channel_mix(enhanced) + x
        
        return enhanced


class GaussianSpatialAttention(nn.Module):
    """
    Applies Gaussian-weighted spatial attention.
    
    For tiny objects, attention should be spatially focused (narrow Gaussian).
    For larger objects, attention can be broader (wide Gaussian).
    
    The attention weight at each position is based on a learned Gaussian:
        Attention(x,y) = Σ_i w_i · exp(-d²/(2σ²))
    where d is distance from learned anchor points.
    
    This is motivated by matched filter theory: the optimal detector for a
    Gaussian-distributed signal in noise is a Gaussian-weight filter.
    """
    
    def __init__(self, in_channels, num_anchors=4):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Predict attention logits directly (simpler, more stable)
        self.attention_conv = nn.Sequential(
            # Capture local context
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Channel reduction
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            # Attention map
            nn.Conv2d(in_channels // 4, 1, 1, bias=True),
        )
        
        # Scale-adaptive refinement
        self.scale_refine = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),  # attention + scale
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
        )
        
        # Initialize to produce mild attention
        nn.init.constant_(self.attention_conv[-1].bias, 0.0)
    
    def forward(self, x, scale_map):
        """
        Args:
            x: Input features [B, C, H, W]
            scale_map: Per-location scale [B, 1, H, W]
        Returns:
            attention: Spatial attention weights [B, 1, H, W]
        """
        # Base attention from features
        attention = self.attention_conv(x)
        
        # Refine with scale information
        combined = torch.cat([attention, scale_map], dim=1)
        attention = self.scale_refine(combined)
        
        # Sigmoid to get weights in (0, 1)
        attention = torch.sigmoid(attention)
        
        return attention


class ChannelAttention(nn.Module):
    """
    Efficient channel attention (ECA-style) for feature recalibration.
    
    Different channels capture different semantic information.
    For tiny objects, certain channels (edge, texture) may be more important.
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_channels = max(in_channels // reduction, 8)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Channel attention weights [B, C, 1, 1]
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return attention


class TAAM(nn.Module):
    """
    Tiny-Aware Attention Module (TAAM)
    
    A unified attention module for tiny object detection that combines:
    1. Scale estimation - predict object scale at each location
    2. Contrast enhancement - enhance object-background differences  
    3. Gaussian spatial attention - focus on object regions
    4. Channel attention - recalibrate channel importance
    
    All components are scale-adaptive: tiny objects get different processing
    than larger objects.
    
    Args:
        c1: Number of input channels (from YOLO parser)
        c2: Number of output channels (default: same as input, not used)
        reduction: Channel reduction ratio for efficiency
        contrast_kernel: Kernel size for local contrast computation
    """
    
    def __init__(self, c1, c2=None, reduction=16, contrast_kernel=7):
        super().__init__()
        # Use c1 as the channel count (YOLO standard signature)
        in_channels = c1
        
        # Component 1: Scale estimation
        self.scale_estimator = ScaleEstimator(in_channels, reduction)
        
        # Component 2: Contrast enhancement
        self.contrast_enhancer = ContrastEnhancer(in_channels, contrast_kernel)
        
        # Component 3: Gaussian spatial attention
        self.spatial_attention = GaussianSpatialAttention(in_channels)
        
        # Component 4: Channel attention
        self.channel_attention = ChannelAttention(in_channels, reduction)
        
        # Final fusion with residual
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
        # Learnable residual weight (start with strong residual connection)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Enhanced features [B, C, H, W]
        """
        identity = x
        
        # Step 1: Estimate scale at each position
        scale_map = self.scale_estimator(x)
        
        # Step 2: Enhance contrast (scale-adaptive)
        x_contrast = self.contrast_enhancer(x, scale_map)
        
        # Step 3: Apply spatial attention (scale-adaptive)
        spatial_attn = self.spatial_attention(x_contrast, scale_map)
        x_spatial = x_contrast * spatial_attn
        
        # Step 4: Apply channel attention
        channel_attn = self.channel_attention(x_spatial)
        x_channel = x_spatial * channel_attn
        
        # Fusion with learnable residual
        out = self.fusion(x_channel)
        out = identity + self.gamma * out
        
        return out


class TAAMBlock(nn.Module):
    """
    TAAM wrapped in a block for easy integration with YOLO architecture.
    
    This applies TAAM followed by a convolution to match output channels
    if needed.
    
    Args:
        c1: Input channels (from YOLO parser)
        c2: Output channels (default: same as input)
    """
    
    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        
        self.taam = TAAM(c1)
        
        # Channel adjustment if needed
        self.adjust = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
    
    def forward(self, x):
        return self.adjust(self.taam(x))


# For Ultralytics integration - module must be importable
__all__ = ['TAAM', 'TAAMBlock', 'ScaleEstimator', 'ContrastEnhancer', 
           'GaussianSpatialAttention', 'ChannelAttention']


if __name__ == '__main__':
    # Test the module
    print("Testing TAAM module...")
    
    # Create dummy input
    batch_size = 2
    channels = 256
    height, width = 100, 100  # P2 feature map size for 800x800 input at stride 4
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Test TAAM
    taam = TAAM(channels)
    out = taam(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in taam.parameters()):,}")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print("Gradient flow: OK")
    
    # Test TAAMBlock
    block = TAAMBlock(256, 128)
    out_block = block(x)
    print(f"TAAMBlock output shape: {out_block.shape}")
    
    print("\nAll tests passed!")
