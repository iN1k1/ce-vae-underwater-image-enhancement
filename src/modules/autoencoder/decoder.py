from typing import Tuple, List

from src.modules.autoencoder.common import *
import numpy as np
import logging

from src.util import rank_zero_log_only

logger = logging.getLogger(__name__)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpConvTranspConvOut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, *, ch: int, out_ch: int, ch_mult: Tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int, attn_resolutions: List[int], dropout: float = 0.0,
                 resamp_with_conv: bool = True, in_channels: int,
                 resolution: int, z_channels: int, give_pre_end: bool = False,
                 long_skip_connection: bool = True,
                 use_capsules: bool = True, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self._use_capsules = use_capsules

        block_in = z_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        if self._use_capsules:

            in_ch_mult = (1,) + tuple(ch_mult)
            block_in = ch * ch_mult[self.num_resolutions - 1]

            # compute in_ch_mult, block_in and curr_res at lowest res
            rank_zero_log_only(logger, f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions")

            # z to block_in
            self.conv_in = torch.nn.Conv2d(z_channels,
                                           block_in,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1)

            # middle
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = AttnBlock(block_in)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

            # upsampling
            self.up = nn.ModuleList()
            for i_level in reversed(range(self.num_resolutions)):
                block = nn.ModuleList()
                attn = nn.ModuleList()
                block_out = ch * ch_mult[i_level]
                for i_block in range(self.num_res_blocks + 1):
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                    block_in = block_out
                    if curr_res in attn_resolutions:
                        attn.append(AttnBlock(block_in))
                up = nn.Module()
                up.block = block
                up.attn = attn
                if i_level != 0:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    curr_res = curr_res * 2
                self.up.insert(0, up)  # prepend to get consistent order

        up_feats = [512, 256, 128, block_in]
        self.long_skip_up_decode = nn.Sequential(
            UpConvTranspConvOut(self.z_shape[1], up_feats[0]),
            UpConvTranspConvOut(up_feats[0], up_feats[1]),
            UpConvTranspConvOut(up_feats[1], up_feats[2]),
            UpConvTranspConvOut(up_feats[2], up_feats[3]),
        ) if long_skip_connection else None

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, enc, z):
        h = 0
        if self._use_capsules:
            self.last_z_shape = z.shape

            # timestep embedding
            temb = None

            # z to block_in
            h = self.conv_in(z)

            # middle
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

            # upsampling
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h = self.up[i_level].block[i_block](h, temb)
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                if i_level != 0:
                    h = self.up[i_level].upsample(h)

            # end
            if self.give_pre_end:
                return h

            h = self.norm_out(h)
            h = nonlinearity(h)

        if self.long_skip_up_decode is not None:
            h = h + self.long_skip_up_decode(enc)
        h = self.conv_out(h)
        return h
