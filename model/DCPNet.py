import torch.nn.functional as F

from model.transformer import Transformer
from model.backbone import Backbone
from torch import nn
import torch


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, high_feat, low_feat):
        x = self.up(high_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


class DCPNet(nn.Module):
    def __init__(self):
        super(DCPNet, self).__init__()

        # Encoder
        backbone = Backbone()
        model_dict = backbone.state_dict()
        trans_path = 'pretrain/pvt_v2_b2.pth'
        trans_save_model = torch.load(trans_path, map_location='cpu')

        cnn_path = 'pretrain/resnet50.pth'
        cnn_save_model = torch.load(cnn_path, map_location='cpu')

        trans_state_dict = {k: v for k, v in trans_save_model.items() if k in model_dict.keys()}
        cnn_state_dict = {k: v for k, v in cnn_save_model.items() if k in model_dict.keys()}
        model_dict.update(trans_state_dict)
        model_dict.update(cnn_state_dict)
        backbone.load_state_dict(model_dict)  # 64, 128 , 320 ,512
        self.backbone = backbone

        # neck模块
        self.transformer = Transformer(512, 2048, 11, 11)

        # Decoder模块
        self.decoder3 = DecoderBlock(512, 320, 320)
        self.decoder2 = DecoderBlock(320, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        self.sal_head = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def upsample(self, x):
        return F.interpolate(x, size=[352, 352], mode='bilinear', align_corners=True)

    def forward(self, x):
        # backbone
        trans_x, cnn_y, out = self.backbone(x)

        # neck
        x = self.transformer(trans_x, cnn_y)

        # Decoder
        x = self.decoder3(x, out[-1])
        x = self.decoder2(x, out[-2])
        x = self.decoder1(x, out[-3])

        output = self.upsample(self.sal_head(x))

        return output, self.sigmoid(output)
