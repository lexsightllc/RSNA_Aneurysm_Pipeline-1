"""Compact 3D CNN model for RSNA Intracranial Aneurysm Detection."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation3D(nn.Module):
    """Squeeze-and-Excitation block for 3D inputs."""
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)


class InvertedResidual3D(nn.Module):
    """Inverted residual block with SE for 3D inputs."""
    def __init__(self, in_channels: int, out_channels: int, stride: int, 
                 expand_ratio: float, use_se: bool = True, 
                 dilation: int = 1):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.Hardswish(inplace=True)
            ])
        
        # Depthwise convolution
        padding = (3 * dilation - dilation) // 2  # For kernel_size=3
        layers.extend([
            nn.Conv3d(
                hidden_dim, hidden_dim, 3, stride, padding, 
                groups=hidden_dim, dilation=dilation, bias=False
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.Hardswish(inplace=True),
        ])
        
        if use_se:
            layers.append(SqueezeExcitation3D(hidden_dim))
            
        # Linear pointwise convolution
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class Compact3DCNN(nn.Module):
    """Compact 3D CNN based on MobileNetV3-Small principles."""
    def __init__(self, num_classes: int = 14, width_mult: float = 0.5):
        super().__init__()
        
        def _make_divisible(v: float, divisor: int = 8) -> int:
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Input conv stem
        input_channels = _make_divisible(16 * width_mult)
        last_channel = _make_divisible(512 * width_mult)
        
        # Building blocks
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv3d(1, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm3d(input_channels),
            nn.Hardswish(inplace=True),
            
            # Inverted residual blocks
            # t, c, n, s, se, dilations
            [1, 16, 1, 1, False, 1],
            [4.5, 24, 2, 2, False, 1],
            [3.7, 40, 2, 2, True, 1],
            [4, 80, 3, 2, False, 1],
            [6, 112, 3, 1, True, 2],  # Dilated for larger receptive field
            [6, 160, 1, 1, True, 2],  # Dilated for larger receptive field
        )
        
        # Build the feature extractor
        input_channel = input_channels
        for t, c, n, s, se, d in self.features:
            output_channel = _make_divisible(c * width_mult)
            layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual3D(
                    input_channel, output_channel, stride, 
                    expand_ratio=t, use_se=se, dilation=d
                ))
                input_channel = output_channel
            self.features.extend(layers)
        
        # Last few layers
        last_conv_channels = _make_divisible(576 * width_mult)
        self.features.append(
            nn.Conv3d(input_channel, last_conv_channels, 1, bias=False)
        )
        self.features.append(nn.BatchNorm3d(last_conv_channels))
        self.features.append(nn.Hardswish(inplace=True))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
            nn.Sigmoid()
        )
        
        # Weight initialization
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 1, D, H, W)
        x = self.features(x)  # (batch, C, D', H', W')
        x = self.avgpool(x)  # (batch, C, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, C)
        x = self.classifier(x)  # (batch, num_classes)
        return x
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
