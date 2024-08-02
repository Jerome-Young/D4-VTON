import torch
import torch.nn as nn
import torch.nn.functional as F
from .ds_selector import ExtractionOperation
from .light_net import LightNet


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)


# backbone
class SementicGroupFlow(nn.Module):
    def __init__(self, in_channels=512, out_channels=2, group_num=8, ks=3, sample_k=6):
        super(SementicGroupFlow, self).__init__()
        self.group_num = group_num
        self.k = sample_k
        layers = []
        self.group_dim = in_channels  # // num_heads
        for i in range(group_num):
            layer = nn.Sequential(
                nn.Conv2d(self.group_dim, 128, kernel_size=ks, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(128, out_channels=64, kernel_size=ks, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(64, out_channels=32, kernel_size=ks, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=ks, stride=1, padding=1)
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, input, last_offsets=None):  # x (B,C,H,W)
        B, C, H, W = input.shape
        offsets = []
        attns = []
        for i in range(self.group_num):
            # global
            feat = input
            offsets_att = self.layers[i](feat)
            attn = offsets_att[:, self.k * 2:, :, :].view(B, 1, 1, H, W)
            offset = apply_offset(offsets_att[:, :self.k * 2, :, :].reshape(-1, 2, H, W))
            if last_offsets is not None:
                offset = F.grid_sample(last_offsets[i], offset, mode='bilinear', padding_mode='border')
            else:
                offset = offset.permute(0, 3, 1, 2)
            offsets.append(offset)
            attns.append(attn)
        attns = torch.cat(attns, dim=1)
        attns = F.softmax(attns, dim=1)
        attns = [attns[:, i] for i in range(self.group_num)]

        return offsets, attns


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64,128,256,256,256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                         ResBlock(out_chns),
                                         ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features


class RefinePyramid(nn.Module):
    def __init__(self, chns=[64,128,256,256,256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


class DSDModules(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256, sample_nums=6, group_nums=8, hidden_dim=64):
        super(DSDModules, self).__init__()
        self.Coarse_Nets = []
        self.Fine_Nets = []
        self.k = sample_nums
        self.g = group_nums
        self.in_ch = 2*fpn_dim
        self.out_ch = fpn_dim
        self.hidden_dim = hidden_dim
        self.Group_Selectors = []
        for i in range(num_pyramid):
            group_selector = ExtractionOperation(in_channel=self.hidden_dim, num_label=self.g)
            Coarse_layer = SementicGroupFlow(in_channels=self.in_ch, out_channels=self.k*3, group_num=self.g, sample_k=self.k)
            Fine_layer = SementicGroupFlow(in_channels=self.in_ch, out_channels=self.k*3, group_num=self.g, sample_k=self.k)
            # group flow
            self.Coarse_Nets.append(Coarse_layer)
            self.Fine_Nets.append(Fine_layer)
            self.Group_Selectors.append(group_selector)

        # group flow
        self.Coarse_Nets = nn.ModuleList(self.Coarse_Nets)
        self.Fine_Nets = nn.ModuleList(self.Fine_Nets)
        self.Group_Selectors = nn.ModuleList(self.Group_Selectors)

        self.lightnet = LightNet()

    def upsample(self, last_offsets, last_attns, num_heads=8):
        cur_offs = []
        cur_attns = []
        for i in range(num_heads):
            cur_off = F.interpolate(last_offsets[i], scale_factor=2, mode='bilinear')
            cur_attn = F.interpolate(last_attns[i], scale_factor=2, mode='bilinear')
            cur_offs.append(cur_off)
            cur_attns.append(cur_attn)
        return cur_offs, cur_attns

    def diswarp(self, input, offset, att_maps, out_ch, group_nums, mask=None):
        B, C, H, W = input.size()
        feats_out = torch.zeros_like(input)
        for i in range(group_nums):
            multi_feat = torch.repeat_interleave(input, self.k, 0)
            att_map = torch.repeat_interleave(att_maps[i], out_ch, 1)
            multi_warp_feat = F.grid_sample(multi_feat, offset[i].permute(0, 2, 3, 1), mode='bilinear',
                                            padding_mode='border')
            multi_att_warp_feat = multi_warp_feat.reshape(B, -1, H, W) * att_map
            warp_feat = sum(torch.split(multi_att_warp_feat, out_ch, 1))
            if mask is not None:
                warp_feat = warp_feat * mask[:, i, :].view(B, C, 1, 1)
            feats_out += warp_feat

        return feats_out

    def flow_estimate(self, gar_feat, cond_feat, mask, last_offsets, num_layer):
        # coarse_warp
        # dynamic group assignment
        input_feat = torch.cat([gar_feat, cond_feat], 1)
        coarse_offsets, coarse_attns = self.Coarse_Nets[num_layer](input_feat, last_offsets=last_offsets)
        coarse_warp_feat = self.diswarp(gar_feat, coarse_offsets, coarse_attns, self.out_ch, self.g, mask=mask)

        # fine_warp
        input_feat = torch.cat([coarse_warp_feat, cond_feat], 1)
        fine_offsets, fine_attns = self.Fine_Nets[num_layer](input_feat, last_offsets=coarse_offsets)

        # Upsampling
        up_fine_offsets, up_fine_attns = self.upsample(fine_offsets, fine_attns, num_heads=self.g)

        return up_fine_offsets, up_fine_attns

    def forward(self, gar_img, gar_feats, cond_feats):
        offsets = None
        results_all = []

        for i in range(len(gar_feats)):
            gar_feat = gar_feats[len(gar_feats) - 1 - i]
            cond_feat = cond_feats[len(cond_feats) - 1 - i]
            B, C, H, W = gar_feat.size()

            cur_gar_img = F.interpolate(gar_img, (H * 2, W * 2), mode='bilinear')
            cur_gar_img = self.lightnet.encoder(cur_gar_img)

            coarse_mask, fine_mask = self.Group_Selectors[i](cur_gar_img)
            offsets, attns = self.flow_estimate(gar_feat, cond_feat, coarse_mask, offsets, i)
            fine_warp = self.diswarp(cur_gar_img, offsets, attns, self.hidden_dim, self.g, mask=fine_mask)

            result_warp = self.lightnet.decoder(fine_warp)

            results_all.append(result_warp)

        return results_all


class DSDNet(nn.Module):
    def __init__(self, cond_in_channel, gar_in_channel=3, sample_nums=6, group_nums=8, hidden_dim=64):
        super(DSDNet, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.garment_features = FeatureEncoder(gar_in_channel, num_filters)
        self.condition_features = FeatureEncoder(cond_in_channel, num_filters)
        self.garment_FPN = RefinePyramid(num_filters)
        self.condition_FPN = RefinePyramid(num_filters)
        self.dsdms = DSDModules(len(num_filters), sample_nums=sample_nums, group_nums=group_nums, hidden_dim=hidden_dim)

    def forward(self, condition, gar_img):
        cond_feats = self.condition_FPN(self.condition_features(condition))
        gar_feats = self.garment_FPN(self.garment_features(gar_img))
        result = self.dsdms(gar_img, gar_feats, cond_feats)

        return result


if __name__ == '__main__':
    net = DSDNet(cond_in_channel=51, sample_nums=1, group_nums=8).cuda()
    # for k,v in net.state_dict().items():
    #     print(k)
    source_image = torch.ones(1,3,512,384).cuda()
    condition = torch.ones(1,51,512,384).cuda()
    out1 = net(condition, source_image)
    print(out1[-1].shape)


