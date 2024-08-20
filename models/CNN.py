"""CNN model"""
from __future__ import print_function
import torch
from torch import nn


class CNN_mass(nn.Module):
    """CNN model that uses either square 64 or 128 pixel images with one additional scalar feature(e.g. mass)"""
    def __init__(self, pixels):
        super().__init__()
        self.pixels = pixels
        conv_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=conv_channels*2,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.AvgPool2d(2, 2)
        )
        self.avgpool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels*2,
                out_channels=conv_channels,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.AvgPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.AvgPool2d(2, 2)
        )
        self.conv5 = nn.Identity()
        if self.pixels == 128:
            self.conv5 = nn.Sequential(
                nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=4,
                    stride=1),
                nn.ReLU(),
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.AvgPool2d(2, 2)

            )

        self.head = nn.Sequential(
            nn.Linear(conv_channels*8*8+1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x_in):
        """Forward pass"""
        x = self.conv1(x_in[0])
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, x_in[1]), dim=1)
        x = self.head(x)
        return x
