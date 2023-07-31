import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.SE import *

'''
U-Net
'''
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, wf=6, padding=True,
                 norm='None', up_mode='upconv', residual=False, dropout=False):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding  # True, 1
        self.depth = depth  # 3 or 4
        self.residual = residual  # False
        self.dropout = dropout

        prev_channels = in_channels  # ic (in_channels)

        self.down_path = nn.ModuleList()   # list for modules
        for i in range(depth): # (1, 2^6, 2^7, 2^8, 2^9)  i =0, 1, 2, 3;  | 16, 8, 4, 2
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf+i), padding, norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):  # 2^9, 2^8, 2^7, 2^6 | i = 2, 1, 0
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf+i), up_mode, padding, norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=1)  # 2^6 to 1

    def forward(self, x, opts_drop):
        input_ = x
        blocks = []
        p_set = [0, 0, 0, 0]  # probability of zero for dropout
        for i, down in enumerate(self.down_path):  # i = 0, 1, 2, 3
            x = down(x)

            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool3d(x, 2)  # Average pooling here, kernel_size = 2
                x = F.dropout(x, p=p_set[i])

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            x = F.dropout(x, p=0.3)

        p_set = [0, 0, 0, 0]
        for i, up in enumerate(self.up_path):  # i = 0, 1, 2
            x = up(x, blocks[-i-1], dropout=p_set[i])

        if self.residual:
            out = input_[:, [0], :, :, :] + self.last(x)  # choose the fisrt channel to residue, while keep the shape
        else:
            out = self.last(x)

        return out     # size = [batch_size, channel, 32, 32, 32]


class UNetConvBlock(nn.Module):  # "Conv3D (+ BN) + ReLU" + "Conv3d (+ BN) + ReLU"
    def __init__(self, in_size, out_size, padding, norm):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv3d(in_size, out_size, kernel_size=3, padding=int(padding)))  # pad = 1
        if norm == 'BN':
            block.append(nn.BatchNorm3d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm3d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if norm == 'BN':
            block.append(nn.BatchNorm3d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm3d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block) # list to module sequential

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width, layer_depth = layer.size()  # 32,32,32
        diff_y = (layer_height - target_size[0]) // 2  # floor division
        diff_x = (layer_width - target_size[1]) // 2
        diff_z = (layer_depth - target_size[2]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1]), diff_z:(diff_z + target_size[2])]

    def forward(self, x, bridge, dropout):
        up = self.up(x)

        crop1 = self.center_crop(bridge, up.shape[2:])

        out = torch.cat([up, crop1], 1)
        out = F.dropout(out, p=dropout)

        out = self.conv_block(out)


        return out

class Weight_Adaptive(nn.Module):
    def __init__(self, n_channels=2, n_filters=32, num_layers=4, growthrate=32, norm='None'):
        super(Weight_Adaptive, self).__init__()
        # Adptive Spatial Weights
        self.input_conv = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB = RDB(n_filters, num_layers, growthrate, norm)
        self.out_conv = nn.Conv3d(n_filters, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid= nn.Sigmoid()

        # Adaptive Channel-wise weights
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(3, 8, bias=True)
        self.fc2 = nn.Linear(8, 3, bias=True)
        self.relu = nn.ReLU()



    def forward(self, input_dc, input_pred, Mask):
        # Feature fusion
        self.input_comb = self.input_conv(torch.cat((input_dc, input_pred), 1))   # 2 to 16
        self.input_extract = self.RDB(self.input_comb)
        self.weight_adap = self.sigmoid(self.out_conv(self.input_extract))  # Should we add a Batch normalization here?)
        # Activation and output
        self.out_dc_1   = input_dc   * Mask * self.weight_adap
        self.out_pred_1 = input_pred * Mask * (1 - self.weight_adap)
        self.out_pred_2 = input_pred * (1 - Mask)

        # Channel-wise recalibration
        self.out_concat = torch.cat((self.out_dc_1, self.out_pred_1, self.out_pred_2), 1)
        batch_size, num_channels, _ , _ , _ = self.out_concat.size()
        self.out_concat_squeeze = self.avg_pool(self.out_concat)

        fc_out_1 = self.relu(self.fc1(self.out_concat_squeeze.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        out_recal = torch.mul(self.out_concat, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        # Channel-wise summation
        out_sum = 2 * torch.sum(out_recal, 1).unsqueeze(1)

        return out_sum


'''
Residual Dense Network
'''
class RDN(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, n_blocks=10, dropout=None):
        super(RDN, self).__init__()
        # F-1
        self.conv1 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1, bias=True)
        self.drop1 = nn.Dropout(p=dropout)
        # RDBs 3
        self.RDBs = nn.ModuleList([RDB(n_filters, n_denselayer, growth_rate) for _ in range(n_blocks)])
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv3d(n_filters*n_blocks, n_filters, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1, bias=True)
        self.drop2 = nn.Dropout(p=dropout)
        # conv
        self.conv3 = nn.Conv3d(n_filters, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        # F_0 = self.drop1(F_0)

        features = []
        x = F_0
        for RDB_ in self.RDBs:
            y = RDB_(x)
            features.append(y)
            x = y
        FF = torch.cat(features, 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        # FDF = self.drop2(FDF)

        output = self.conv3(FDF)

        return output


# Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)


    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x # Residual
        return out



'''
spatial-channel Squeeze and Excite Residual Dense UNet (depth = 4)
'''
class scSERDUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None', dropout=False):
        super(scSERDUNet, self).__init__()
        self.dropout = dropout
        self.conv_in = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB4 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE4 = ChannelSpatialSELayer3D(n_filters * 1, norm='None')

        # decode
        self.up3 = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.fuse_up3 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up3 = RDB(n_filters * 1, n_denselayer, growth_rate, norm)
        self.SE_up3 = ChannelSpatialSELayer3D(n_filters * 1, norm='None')

        self.up2 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up2 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up1 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x, opts_drop):
        # encode
        down1 = self.conv_in(x)
        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)

        down2 = F.avg_pool3d(SE1, 2)
        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)

        down3 = F.avg_pool3d(SE2, 2)
        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)

        down4 = F.avg_pool3d(SE3, 2)
        RDB4 = self.RDB4(down4)
        SE4 = self.SE4(RDB4)

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            SE4 = F.dropout(SE4, p=0.3)

        # decode
        up3 = self.up3(SE4) # ([2, 64, 18, 18, 10])
        RDB_up3 = self.RDB_up3(self.fuse_up3(torch.cat((up3, SE3), 1)))
        SE_up3 = self.SE_up3(RDB_up3)

        up2 = self.up2(SE_up3)
        RDB_up2 = self.RDB_up2(self.fuse_up2(torch.cat((up2, SE2), 1)))
        SE_up2 = self.SE_up2(RDB_up2)

        up1 = self.up1(SE_up2)
        RDB_up1 = self.RDB_up1(self.fuse_up1(torch.cat((up1, SE1), 1)))
        SE_up1 = self.SE_up1(RDB_up1)

        output = self.conv_out(SE_up1)
        return output




'''
spatial-channel Squeeze and Excite Residual Dense UNet (depth = 3)
'''
class scSERDUNet3(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None', dropout=False):
        super(scSERDUNet3, self).__init__()

        self.dropout = dropout
        self.conv_in = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        # decode
        self.up2 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up2 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up1 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x, opts_drop):
        # encode
        down1 = self.conv_in(x)
        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)

        down2 = F.avg_pool3d(SE1, 2)
        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)

        down3 = F.avg_pool3d(SE2, 2)
        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            SE3 = F.dropout(SE3, p=0.3)

        # decode
        up2 = self.up2(SE3)
        RDB_up2 = self.RDB_up2(self.fuse_up2(torch.cat((up2, SE2), 1)))
        SE_up2 = self.SE_up2(RDB_up2)

        up1 = self.up1(SE_up2)
        RDB_up1 = self.RDB_up1(self.fuse_up1(torch.cat((up1, SE1), 1)))
        SE_up1 = self.SE_up1(RDB_up1)

        output = self.conv_out(SE_up1)

        return output



def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


if __name__ == '__main__':
    pass

