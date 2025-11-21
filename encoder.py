import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # 12×50 的 HMOG 片段上保留更多空间分辨率，避免把 12 个传感器维度压到过低的 3 行
        channels = [128, 192, 256, 320]
        attn_resolutions = []  # 依赖瓶颈处的 NonLocal 即可
        num_res_blocks = 2
        resolution = 12  # 粗略跟踪特征维度高度
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        downsample_plan = [
            {"stride": (2, 2), "pad": (0, 1, 0, 1)},  # 12x50 -> 6x25
            {"stride": (1, 2), "pad": (0, 1, 1, 1)},  # 6x25 -> 6x12（仅压缩时间轴）
            {"stride": (1, 2), "pad": (0, 1, 1, 1)},  # 6x12 -> 6x6  得到 36 个 token
        ]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i < len(downsample_plan):
                cfg = downsample_plan[i]
                layers.append(DownSampleBlock(channels[i + 1], stride=cfg["stride"], pad=cfg["pad"]))
                # 粗略跟踪分辨率，主要用于是否添加 NonLocalBlock
                resolution = resolution // cfg["stride"][0]
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
