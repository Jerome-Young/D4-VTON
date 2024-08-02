import torch
import math
import sys
from torch import nn


class ExtractionOperation(nn.Module):
    def __init__(self, in_channel=256, out_channel=256, num_label=8, match_kernel=3):
        super(ExtractionOperation, self).__init__()
        self.value_conv = EqualConv2d(in_channel, in_channel, match_kernel, 1, match_kernel//2, bias=True)
        self.semantic_extraction_filter = EqualConv2d(in_channel, num_label, match_kernel, 1, match_kernel//2, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.num_label = num_label
        self.proj = nn.Linear(in_channel, out_channel)

    def forward(self, value):
        key = value
        b, c, h, w = value.shape
        key = self.semantic_extraction_filter(self.feature_norm(key))
        extraction_softmax = key.view(b, -1, h*w)  # bkm
        values_flatten = self.value_conv(value).view(b, -1, h*w)
        neural_textures = torch.einsum('bkm,bvm->bkv', extraction_softmax, values_flatten)
        attn = self.proj(neural_textures)
        coarse_mask = gumbel_softmax(attn)
        fine_mask = gumbel_softmax(neural_textures)

        return coarse_mask, fine_mask  # extraction_softmax

    def feature_norm(self, input_tensor):
        input_tensor = input_tensor - input_tensor.mean(dim=1, keepdim=True)
        norm = torch.norm(input_tensor, 2, 1, keepdim=True) + sys.float_info.epsilon
        out = torch.div(input_tensor, norm)
        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = torch.nn.functional.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, dim: int = -2) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


if __name__ == '__main__':
    net = ExtractionOperation(64, num_label=8, match_kernel=3).cuda()
    # for k,v in net.state_dict().items():
    #     print(k)
    garment = torch.ones(2, 64, 256, 192).cuda()
    mask1, mask2 = net(garment)
    print(mask1.shape)
