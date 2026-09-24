import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
import einops
import numpy as np
import math


class NICE_Conv3D(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 enc_channels: int = 8,
                 dec_channels: int = 16,
                 decoder_layer: int = 3,
                 kernel_size: int = 3,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Encoder = Conv_encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint=use_checkpoint)
        if decoder_layer == 1:
            self.Decoder = Conv_decoder_Swin1(in_channels=enc_channels,
                                    channel_num=dec_channels,
                                    use_checkpoint=use_checkpoint)
        elif decoder_layer == 2:
            self.Decoder = Conv_decoder_Swin2(in_channels=enc_channels,
                                              channel_num=dec_channels,
                                              use_checkpoint=use_checkpoint)
        elif decoder_layer == 4:
            self.Decoder = Conv_decoder_Swin4(in_channels=enc_channels,
                                              channel_num=dec_channels,
                                              use_checkpoint=use_checkpoint)
        elif decoder_layer == 11:
            self.Decoder = Conv_decoder_CNN1(in_channels=enc_channels,
                                             channel_num=dec_channels,
                                             use_checkpoint=use_checkpoint)
        elif decoder_layer == 12:
            self.Decoder = Conv_decoder_CNN2(in_channels=enc_channels,
                                             channel_num=dec_channels,
                                             use_checkpoint=use_checkpoint)
        elif decoder_layer == 13:
            self.Decoder = Conv_decoder_CNN(in_channels=enc_channels,
                                        channel_num=dec_channels,
                                        use_checkpoint=use_checkpoint)
        elif decoder_layer == 14:
            self.Decoder = Conv_decoder_CNN4(in_channels=enc_channels,
                                             channel_num=dec_channels,
                                             use_checkpoint=use_checkpoint)
        elif decoder_layer == 23:
            self.Decoder = Conv_decoder_CNN_Layer2(in_channels=enc_channels,
                                                    channel_num=dec_channels,
                                                    use_checkpoint=use_checkpoint)
        elif decoder_layer == 22:
            self.Decoder = Conv_decoder_CNN_Layer2_2(in_channels=enc_channels,
                                                    channel_num=dec_channels,
                                                    use_checkpoint=use_checkpoint)

        elif decoder_layer == 33:
            self.Decoder = Conv_decoder_CNN_Layer1(in_channels=enc_channels,
                                                    channel_num=dec_channels,
                                                    use_checkpoint=use_checkpoint)
        else:
            self.Decoder = Conv_decoder(in_channels=enc_channels,
                                        channel_num=dec_channels,
                                        kernel_size = kernel_size,
                                        use_checkpoint=use_checkpoint)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, fixed, moving):
        x_fix = self.Encoder(fixed)
        x_mov = self.Encoder(moving)
        flow = self.Decoder(x_fix, x_mov)
        warped = self.SpatialTransformer(moving, flow)

        return [warped], [flow]  # , affined, affine_para


########################################################
# Encoder/Decoder
########################################################
class Conv_encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.conv_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.conv_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.conv_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)

        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.conv_1(x_in)

        x = self.downsample(x_1)
        x_2 = self.conv_2(x)

        x = self.downsample(x_2)
        x_3 = self.conv_3(x)

        return [x_1, x_2, x_3]


class Conv_decoder(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 kernel_size: int = 3,
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        self.swin_tf_1 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_5 = DeformHead_block(bot_channels, kernel_size=kernel_size, use_checkpoint=use_checkpoint)

        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_1 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)
        self.swin_tf_2 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, kernel_size=kernel_size, use_checkpoint=use_checkpoint)

        self.sample_2 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)
        self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, kernel_size=kernel_size, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, kernel_size=kernel_size, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, kernel_size=kernel_size, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_5 = self.swin_tf_1(x)
        flow_5 = self.reghead_5(x_5)
        x_mov_5_warped = self.SpatialTransformer(x_mov_3, flow_5)

        # Swin decoder Step 2 (Bottleneck 迭代细化)
        x = torch.cat([x_5, x_fix_3, x_mov_5_warped], dim=1)
        x = self.sample_1(x)
        x_4 = self.swin_tf_2(x)
        x = self.reghead_4(x_4)
        flow_4 = x + flow_5  # 现在空间大小一致了，可以安全相加
        x_mov_4_warped = self.SpatialTransformer(x_mov_3, flow_4)

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_4, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_2(x)
        x_3 = self.swin_tf_3(x)
        x = self.reghead_3(x_3)
        flow_3 = x + flow_4

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_CNN(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        """self.swin_tf_1 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_1 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_5 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_1 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)
        """self.swin_tf_2 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_2 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        self.sample_2 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)
        """self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_3 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_5 = self.swin_tf_1(x)
        flow_5 = self.reghead_5(x_5)
        x_mov_5_warped = self.SpatialTransformer(x_mov_3, flow_5)

        # Swin decoder Step 2 (Bottleneck 迭代细化)
        x = torch.cat([x_5, x_fix_3, x_mov_5_warped], dim=1)
        x = self.sample_1(x)
        x_4 = self.swin_tf_2(x)
        x = self.reghead_4(x_4)
        flow_4 = x + flow_5  # 现在空间大小一致了，可以安全相加
        x_mov_4_warped = self.SpatialTransformer(x_mov_3, flow_4)

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_4, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_2(x)
        x_3 = self.swin_tf_3(x)
        x = self.reghead_3(x_3)
        flow_3 = x + flow_4

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_CNN_Layer2(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        self.swin_tf_3 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        """"""
        self.swin_tf_1 = Conv_block(in_channels * 4 + channel_num * 2, in_channels * 4 + channel_num * 2, use_checkpoint)
        self.reghead_5 = DeformHead_block(in_channels * 4 + channel_num * 2, use_checkpoint=use_checkpoint)

        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_1 = nn.Conv3d(in_channels * 4 + channel_num * 2, in_channels * 2 + channel_num * 1, kernel_size=1)
        """self.swin_tf_2 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.swin_tf_1 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_2 = Conv_block(in_channels * 4 + channel_num * 2, in_channels * 4 + channel_num * 2, use_checkpoint)
        self.reghead_4 = DeformHead_block(in_channels * 4 + channel_num * 2, use_checkpoint=use_checkpoint)

        self.sample_2 = nn.Conv3d(in_channels * 4 + channel_num * 2, in_channels * 2 + channel_num * 1, kernel_size=1)
        self.sample_3 = nn.Conv3d(in_channels * 8 + channel_num * 4, in_channels * 4 + channel_num * 2, kernel_size=1)
        self.dec_conv_2 = Conv_block(in_channels * 4 + channel_num * 2, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_3 = self.swin_tf_3(x)
        flow_3 = self.reghead_3(x_3)

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        # layer 2-1
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        #
        x_2_1 = self.swin_tf_1(x)
        x = self.reghead_5(x_2_1)
        flow_2_1 = x + flow_3_up

        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_2_1)
        x_2_1 = self.sample_1(x_2_1)
        x = torch.cat([x_2_1, x_fix_2, x_mov_2_warped], dim=1)
        # x = self.sample_2(x)
        x_2_2 = self.swin_tf_2(x)
        x = self.reghead_4(x_2_2)
        flow_2_2 = x + flow_2_1

        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_2_2)
        x_2_2 = self.sample_2(x_2_2)
        x = torch.cat([x_2_2, x_fix_2, x_mov_2_warped], dim=1)
        # x = self.sample_3(x)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_2_2

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_CNN_Layer2_2(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        self.swin_tf_3 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        """"""
        # self.sample_1 = nn.Conv3d(in_channels * 2 + channel_num * 1, in_channels * 2 + channel_num * 1, kernel_size=1)
        self.swin_tf_1 = Conv_block(in_channels * 4 + channel_num * 2, in_channels * 4 + channel_num * 2,
                                    use_checkpoint)
        self.reghead_5 = DeformHead_block(in_channels * 4 + channel_num * 2, use_checkpoint=use_checkpoint)

        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_2 = nn.Conv3d(in_channels * 8 + channel_num * 2, in_channels * 4 + channel_num * 2, kernel_size=1)
        self.swin_tf_2 = Conv_block(in_channels * 4 + channel_num * 2, in_channels * 4 + channel_num * 2,
                                    use_checkpoint)
        self.reghead_4 = DeformHead_block(in_channels * 4 + channel_num * 2, use_checkpoint=use_checkpoint)

        self.sample_3 = nn.Conv3d(in_channels * 8 + channel_num * 2, in_channels * 4 + channel_num * 2, kernel_size=1)

        self.dec_conv_2 = Conv_block(in_channels * 4 + channel_num * 2, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_3 = self.swin_tf_3(x)
        flow_3 = self.reghead_3(x_3)

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_3_up = self.upsample_2(x_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)
        x = torch.cat([x_3_up, x_fix_2, x_mov_2_warped], dim=1)

        # x = self.sample_1(x)
        x_2_1 = self.swin_tf_1(x)
        x = self.reghead_5(x_2_1)
        flow_2_1 = x + flow_3_up
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_2_1)
        x = torch.cat([x_2_1, x_fix_2, x_mov_2_warped], dim=1)
        x = self.sample_2(x)
        x_2_2 = self.swin_tf_2(x)
        x = self.reghead_4(x_2_2)
        flow_2_2 = x + flow_2_1

        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_2_2)
        x = torch.cat([x_2_2, x_fix_2, x_mov_2_warped], dim=1)
        x = self.sample_3(x)

        x_2 = self.dec_conv_2(x)
        x = self.reghead_2(x_2)
        flow_2 = x + flow_2_2

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_2_up = self.upsample_1(x_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)
        x = torch.cat([x_2_up, x_fix_1, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1

class Conv_decoder_CNN_Layer1(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        self.swin_tf_3 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        """"""
        self.dec_conv_2 = Conv_block(in_channels * 4 + channel_num * 2, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)

        # self.sample_1 = nn.Conv3d(in_channels * 2 + channel_num * 1, in_channels * 2 + channel_num * 1, kernel_size=1)
        self.swin_tf_1 = Conv_block(in_channels * 2 + channel_num * 1, in_channels * 2 + channel_num * 1,
                                    use_checkpoint)
        self.reghead_5 = DeformHead_block(in_channels * 2 + channel_num * 1, use_checkpoint=use_checkpoint)

        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_2 = nn.Conv3d(in_channels * 4 + channel_num * 1, in_channels * 2 + channel_num * 1, kernel_size=1)
        self.swin_tf_2 = Conv_block(in_channels * 2 + channel_num * 1, in_channels * 2 + channel_num * 1,
                                    use_checkpoint)
        self.reghead_4 = DeformHead_block(in_channels * 2 + channel_num * 1, use_checkpoint=use_checkpoint)

        self.sample_3 = nn.Conv3d(in_channels * 4 + channel_num * 1, in_channels * 2 + channel_num * 1, kernel_size=1)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_3 = self.swin_tf_3(x)
        flow_3 = self.reghead_3(x_3)

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_3_up = self.upsample_2(x_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)
        x = torch.cat([x_3_up, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_2_up = self.upsample_1(x_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)
        x = torch.cat([x_2_up, x_fix_1, x_mov_1_warped], dim=1)
        # x = self.sample_1(x)
        x_1_1 = self.swin_tf_1(x)
        x = self.reghead_5(x_1_1)

        flow_1_1 = x + flow_2_up
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_1_1)
        x = torch.cat([x_1_1, x_fix_1, x_mov_1_warped], dim=1)
        x = self.sample_2(x)
        x_1_2 = self.swin_tf_2(x)
        x = self.reghead_4(x_1_2)
        flow_1_2 = x + flow_1_1

        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_1_2)
        x = torch.cat([x_1_2, x_fix_1, x_mov_1_warped], dim=1)
        x = self.sample_3(x)

        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_1_2

        return flow_1

class Conv_decoder_Swin4(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        self.swin_tf_1 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_5 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_1 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)

        self.swin_tf_2 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        self.sample_2 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)

        self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        self.sample_3 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)

        self.swin_tf_4 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_3_1 = self.swin_tf_1(x)
        flow_3_1 = self.reghead_5(x_3_1)
        x_mov_5_warped = self.SpatialTransformer(x_mov_3, flow_3_1)

        # Swin decoder Step 2 (Bottleneck 迭代细化)
        x = torch.cat([x_3_1, x_fix_3, x_mov_5_warped], dim=1)
        x = self.sample_1(x)
        x_3_2 = self.swin_tf_2(x)
        x = self.reghead_4(x_3_2)
        flow_3_2 = x + flow_3_1  # 现在空间大小一致了，可以安全相加
        x_mov_4_warped = self.SpatialTransformer(x_mov_3, flow_3_2)

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_3_2, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_2(x)
        x_3_3 = self.swin_tf_3(x)
        x = self.reghead_3(x_3_3)
        flow_3_3 = x + flow_3_2

        # Swin decoder Step 4 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_3_3, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_3(x)
        x_3_4 = self.swin_tf_4(x)
        x = self.reghead_3(x_3_4)
        flow_3 = x + flow_3_3

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3_4)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_CNN4(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        """self.swin_tf_1 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_1 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_5 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_1 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)

        """self.swin_tf_2 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_2 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        self.sample_2 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)

        """self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_3 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        self.sample_3 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)

        """self.swin_tf_4 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_4 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)
        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_3_1 = self.swin_tf_1(x)
        flow_3_1 = self.reghead_5(x_3_1)
        x_mov_5_warped = self.SpatialTransformer(x_mov_3, flow_3_1)

        # Swin decoder Step 2 (Bottleneck 迭代细化)
        x = torch.cat([x_3_1, x_fix_3, x_mov_5_warped], dim=1)
        x = self.sample_1(x)
        x_3_2 = self.swin_tf_2(x)
        x = self.reghead_4(x_3_2)
        flow_3_2 = x + flow_3_1  # 现在空间大小一致了，可以安全相加
        x_mov_4_warped = self.SpatialTransformer(x_mov_3, flow_3_2)

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_3_2, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_2(x)
        x_3_3 = self.swin_tf_3(x)
        x = self.reghead_3(x_3_3)
        flow_3_3 = x + flow_3_2

        # Swin decoder Step 4 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_3_3, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_3(x)
        x_3_4 = self.swin_tf_4(x)
        x = self.reghead_3(x_3_4)
        flow_3 = x + flow_3_3

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3_4)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_Swin2(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        """self.swin_tf_1 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_5 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # 使用 1x1 卷积代替 PatchExpanding_block，目的是只降通道，不放大空间！
        self.sample_1 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)"""
        self.swin_tf_2 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        self.sample_2 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)
        self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_4 = self.swin_tf_2(x)
        flow_4 = self.reghead_4(x_4)
        # flow_4 = x + flow_5  # 现在空间大小一致了，可以安全相加
        x_mov_4_warped = self.SpatialTransformer(x_mov_3, flow_4)

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_4, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_2(x)
        x_3 = self.swin_tf_3(x)
        x = self.reghead_3(x_3)
        flow_3 = x + flow_4

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_CNN2(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        self.swin_tf_2 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_4 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        self.sample_2 = nn.Conv3d(bot_channels + in_channels * 8, bot_channels, kernel_size=1)
        self.swin_tf_3 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_4 = self.swin_tf_2(x)
        flow_4 = self.reghead_4(x_4)
        # flow_4 = x + flow_5  # 现在空间大小一致了，可以安全相加
        x_mov_4_warped = self.SpatialTransformer(x_mov_3, flow_4)

        # Swin decoder Step 3 (Bottleneck 最终细化)
        # 修复原代码中的逻辑错误：应该传入 x_4 的特征，而不是原始图像
        x = torch.cat([x_4, x_fix_3, x_mov_4_warped], dim=1)
        x = self.sample_2(x)
        x_3 = self.swin_tf_3(x)
        x = self.reghead_3(x_3)
        flow_3 = x + flow_4

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_Swin1(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_3 = self.swin_tf_3(x)
        flow_3 = self.reghead_3(x_3)
        # flow_3 = x + flow_4

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class Conv_decoder_CNN1(nn.Module):
    def __init__(self,
                 in_channels: int,  # 对应 enc_channels (默认8)
                 channel_num: int,  # 对应 dec_channels (默认16)
                 use_checkpoint: bool = False):
        super().__init__()

        bot_channels = in_channels * 8  # 最底层 Bottleneck 的通道数 (64)

        # ==========================================
        # Level 3 (最底层 Bottleneck: 8x8x8)
        # ==========================================
        """self.swin_tf_3 = SwinTrans_stage_block(embed_dim=bot_channels, num_layers=4,
                                               num_heads=in_channels, window_size=[5, 5, 5],
                                               use_checkpoint=use_checkpoint)"""
        self.swin_tf_3 = Conv_block(in_channels * 8, in_channels * 8, use_checkpoint)
        self.reghead_3 = DeformHead_block(bot_channels, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 2 (中层: 16x16x16)
        # ==========================================
        # 补齐缺失的上采样层
        self.upsample_2 = nn.ConvTranspose3d(bot_channels, channel_num * 2, kernel_size=2, stride=2)
        # 修复拼接后的通道数 (32 + 16 + 16 = 64 = channel_num * 4)
        self.dec_conv_2 = Conv_block(channel_num * 4, channel_num * 2, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint=use_checkpoint)

        # ==========================================
        # Level 1 (顶层: 32x32x32)
        # ==========================================
        self.upsample_1 = nn.ConvTranspose3d(channel_num * 2, channel_num, kernel_size=2, stride=2)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint=use_checkpoint)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3 = x_fix
        x_mov_1, x_mov_2, x_mov_3 = x_mov

        # Swin decoder Step 1 (Bottleneck)
        x = torch.cat([x_fix_3, x_mov_3], dim=1)
        x_3 = self.swin_tf_3(x)
        flow_3 = self.reghead_3(x_3)
        # flow_3 = x + flow_4

        # Step 2 (Level 2: 空间上采样)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.upsample_2(x_3)
        x = torch.cat([x, x_fix_2, x_mov_2_warped], dim=1)
        x_2 = self.dec_conv_2(x)

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 3 (Level 1: 回到原图尺寸)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1_warped], dim=1)
        x_1 = self.conv_1(x)

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1

########################################################
# Blocks (无修改，保持原样即可)
########################################################

class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x


class Conv_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.InstanceNorm3d(out_channels)

        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.InstanceNorm3d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)
        return x_out


class DeformHead_block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3,
                 downsample_stride: int = 2,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        Conv = getattr(nn, 'Conv%dd' % 3)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % 3)
        # downsample 1
        # self.downsample = nn.AvgPool3d(downsample_stride, stride=downsample_stride)
        self.block = torch.nn.Sequential(
            Conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=in_channels // 2, padding=1),
            BatchNorm(in_channels // 2),
            torch.nn.ReLU(),
            # 最后一层直接输出 3 个通道 (分别代表 x, y, z 轴的位移)
            Conv(kernel_size=kernel_size, in_channels=in_channels // 2, out_channels=3, padding=1),
        )

    def forward(self, x_in):
        x = x_in.contiguous()
        # x = self.downsample(x)
        if self.use_checkpoint and x.requires_grad:
            x_out = checkpoint.checkpoint(self.block, x)
        else:
            x_out = self.block(x)

        return x_out


class DeformHead_GridBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int=3,
                 downsample_stride: int=2,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        Conv = getattr(nn, 'Conv%dd' % 3)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % 3)
        # downsample 1
        self.downsample = nn.AvgPool3d(downsample_stride, stride=downsample_stride)
        self.block = torch.nn.Sequential(
            Conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=in_channels // 2, padding=1),
            BatchNorm(in_channels // 2),
            torch.nn.ReLU(),
            # 最后一层直接输出 3 个通道 (分别代表 x, y, z 轴的位移)
            Conv(kernel_size=kernel_size, in_channels=in_channels // 2, out_channels=3, padding=1),
        )
        # downsample 2
        """self.block = torch.nn.Sequential(
            # 将 stride 放在第一层卷积中进行下采样
            # padding 取决于 kernel_size，通常设为 (kernel_size - 1) // 2 即可，但遇到大 stride 时可能需要仔细计算边缘
            Conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=in_channels // 2,
                 stride=downsample_stride, padding=1),
            BatchNorm(in_channels // 2),
            torch.nn.ReLU(),
            # 输出控制点
            Conv(kernel_size=kernel_size, in_channels=in_channels // 2, out_channels=3, padding=1),
        )"""

    def forward(self, x_in):
        x = x_in.contiguous()
        x = self.downsample(x)
        if self.use_checkpoint and x.requires_grad:
            x_out = checkpoint.checkpoint(self.block, x)
        else:
            x_out = self.block(x)

        return x_out


class BSplineDDFGenerator(nn.Module):
    """
    3D B-Spline 密集形变场生成器 (适配 5D Tensor)
    接收: (B, C, Z_ctrl, Y_ctrl, X_ctrl) -> 例如 (2, 3, 16, 32, 32)
    输出: (B, C, Z_img, Y_img, X_img) -> 放大后的密集形变场
    """

    def __init__(self, grid_size=(16, 32, 32), image_size=(128, 256, 256)):
        super().__init__()
        self.grid_size = grid_size
        self.image_size = image_size

        # 预计算三个维度的 1D B-Spline 权重矩阵
        # 形状分别为: (Z_img, Z_ctrl), (Y_img, Y_ctrl), (X_img, X_ctrl)
        W_z = self._build_1d_bspline_matrix(grid_size[0], image_size[0])
        W_y = self._build_1d_bspline_matrix(grid_size[1], image_size[1])
        W_x = self._build_1d_bspline_matrix(grid_size[2], image_size[2])

        # 注册为 buffer，这样它们会随着模型自动保存在 GPU 上，且不会计算梯度
        self.register_buffer('W_z', W_z)
        self.register_buffer('W_y', W_y)
        self.register_buffer('W_x', W_x)

    def _build_1d_bspline_matrix(self, num_ctrl_pts, out_size):
        W = np.zeros((out_size, num_ctrl_pts), dtype=np.float32)
        stride = (out_size - 1) / (num_ctrl_pts - 1)

        def B0(u):
            return (1 - u) ** 3 / 6.0

        def B1(u):
            return (3 * u ** 3 - 6 * u ** 2 + 4) / 6.0

        def B2(u):
            return (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6.0

        def B3(u):
            return u ** 3 / 6.0

        for p in range(out_size):
            idx = p / stride
            i = math.floor(idx)
            u = idx - i

            if i - 1 >= 0:                  W[p, i - 1] += B0(u)
            if i >= 0 and i < num_ctrl_pts: W[p, i] += B1(u)
            if i + 1 < num_ctrl_pts:        W[p, i + 1] += B2(u)
            if i + 2 < num_ctrl_pts:        W[p, i + 2] += B3(u)

            row_sum = np.sum(W[p, :])
            if row_sum > 0:
                W[p, :] /= row_sum

        return torch.tensor(W, dtype=torch.float32)

    def forward(self, ctrl_pts_displacement):
        """
        参数:
            ctrl_pts_displacement: 5D Tensor (B, C, Z_ctrl, Y_ctrl, X_ctrl)
                                   例如 (2, 3, 16, 32, 32)
        返回:
            ddf: 放大后的密集形变场 (B, C, Z_img, Y_img, X_img)
        """
        # =======================================================
        # 核心修复：使用三步张量收缩，逐维度放大网格，节省显存
        # 字母含义:
        # b=Batch, c=Channel
        # i,j,k = 控制点 Z, Y, X 维度
        # z,y,x = 原图 Z, Y, X 维度
        # =======================================================

        # 1. 仅在 Z 轴方向插值放大: (B, C, Z_ctrl, Y_ctrl, X_ctrl) -> (B, C, Z_img, Y_ctrl, X_ctrl)
        step_z = torch.einsum('zi, bcijk -> bczjk', self.W_z, ctrl_pts_displacement)

        # 2. 仅在 Y 轴方向插值放大: (B, C, Z_img, Y_ctrl, X_ctrl) -> (B, C, Z_img, Y_img, X_ctrl)
        step_y = torch.einsum('yj, bczjk -> bczyk', self.W_y, step_z)

        # 释放无用内存
        del step_z

        # 3. 仅在 X 轴方向插值放大: (B, C, Z_img, Y_img, X_ctrl) -> (B, C, Z_img, Y_img, X_img)
        ddf = torch.einsum('xk, bczyk -> bczyx', self.W_x, step_y)

        # 释放无用内存
        del step_y

        return ddf


class PatchExpanding_block(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x_in):
        x = self.up_conv(x_in)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out



class SwinTrans_stage_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    window_size=self.window_size,
                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    use_checkpoint=use_checkpoint)
            self.blocks.append(block)

    def forward(self, x_in):

        b, c, d, h, w = x_in.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)

        x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out


class SwinTrans_Block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 shift_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MSA_block(embed_dim,
                              window_size=window_size,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(hidden_size=embed_dim,
                             mlp_dim=int(embed_dim * mlp_ratio),
                             dropout_rate=drop)

    def forward_part1(self, x_in, mask_matrix):

        x = self.norm1(x_in)

        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):

        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, mask_matrix=None):

        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)

        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x_out = x + self.forward_part2(x)

        return x_out


class MSA_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) *
                                                                     (2 * self.window_size[1] - 1) *
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, mask=None):

        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x_out = self.proj_drop(x)

        return x_out


class MLP_block(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.GELU = nn.GELU()

    def forward(self, x_in):
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x_out = self.drop2(x)

        return x_out


########################################################
# Functions
########################################################

def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition(x_in, window_size):
    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  c)

    return windows


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def test_NICE_Conv3D():
    print("--- 开始测试 NICE_Conv3D 模型 ---")

    # 1. 设置设备 (优先使用 GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # 2. 实例化模型
    # in_channels=1 代表单通道的 3D 图像 (如普通的 CT 或 MRI)
    model = NICE_Conv3D(in_channels=1, enc_channels=8, dec_channels=16, decoder_layer=22, use_checkpoint=False)
    model = model.to(device)
    model.train()  # 设置为训练模式以测试梯度

    # 3. 创建伪造的 3D 图像张量 (Dummy Data)
    # 尺寸格式: (Batch_size, Channels, Depth, Height, Width)
    # 注意: D, H, W 最好是 8 的倍数，因为 Encoder 中有 3 次池化 (2^3 = 8)
    B, C, D, H, W = 1, 1, 32, 128, 128
    fixed_image = torch.randn(B, C, D, H, W).to(device)
    moving_image = torch.randn(B, C, D, H, W).to(device)

    print(f"\n[输入尺寸]")
    print(f"Fixed image shape:  {fixed_image.shape}")
    print(f"Moving image shape: {moving_image.shape}")

    try:
        # 4. 测试前向传播
        # 根据你的 forward 函数，它返回的是列表: [warped], [flow]
        warped_list, flow_list = model(fixed_image, moving_image)

        warped_image = warped_list[0]
        flow_field = flow_list[0]

        print(f"\n[前向传播成功]")
        print(f"Warped image shape: {warped_image.shape} (应当与输入图像尺寸一致)")
        print(f"Flow field shape:   {flow_field.shape} (应当为 B, 3, D, H, W)")

        # 验证尺寸是否符合预期
        assert warped_image.shape == fixed_image.shape, "Warped 图像尺寸错误！"
        assert flow_field.shape == (B, 3, D, H, W), "Flow 场尺寸错误！应该包含 3 个通道 (x,y,z 位移)"

        # 5. 测试反向传播 (梯度计算)
        # 用 sum() 模拟一个假的 Loss
        loss = warped_image.sum() + flow_field.sum()
        loss.backward()

        print(f"\n[反向传播成功]")
        print("计算图完整，梯度可以正常回传！")

        # 打印一下模型参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型总可训练参数量: {total_params:,}")

    except Exception as e:
        print(f"\n[测试失败] 出现错误: {e}")


if __name__ == "__main__":
    test_NICE_Conv3D()