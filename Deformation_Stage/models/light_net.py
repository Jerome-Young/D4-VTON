import torch
import torch.nn as nn
import torch.nn.functional as F


class LightNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super(LightNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channels=self.hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=self.hidden_dim // 2, out_channels=self.hidden_dim, kernel_size=1, stride=1)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(self.hidden_dim, out_channels=self.hidden_dim // 2, kernel_size=1, stride=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=self.hidden_dim // 2, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, gar_img, num_layer):
        results = []
        for num in range(num_layer):
            cur_gar_img = F.interpolate(gar_img, scale_factor=0.5 ** (4 - num), mode='bilinear')
            x = self.encoder(cur_gar_img)
            x = self.decoder(x)
            results.append(x)
        return results