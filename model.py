import torch
import torch.nn as nn
import numpy as np
import math
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.8, stride=1, padding=1, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, math.floor(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.bn_h = norm_layer(math.ceil(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(math.floor(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)
        x_h = self.relu(self.bn_h(X_h))
        x_l = self.relu(self.bn_l(X_l))
        return x_h, x_l
    
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.8, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.stride = stride
        self.l2l = torch.nn.Conv2d(math.floor(alpha * in_channels), math.floor(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(math.floor(alpha * in_channels), out_channels - math.floor(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - math.floor(alpha * in_channels), math.floor(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - math.floor(alpha * in_channels),
                                   out_channels - math.floor(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.bn_h = norm_layer(math.ceil(out_channels*(1-alpha)))
        self.bn_l = norm_layer(math.floor(out_channels*alpha))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)
        
        X_l2h = nn.functional.interpolate(X_l2h, X_h2h.size()[2:])

        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l
        x_h = self.relu(self.bn_h(X_h))
        x_l = self.relu(self.bn_l(X_l))
        return x_h, x_l    
    
class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.8, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.l2h = torch.nn.Conv2d(math.floor(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - math.floor(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = nn.functional.interpolate(X_l2h, X_h2h.size()[2:])
        
        X_h = X_h2h + X_l2h
        x_h = self.relu(self.bn_h(X_h))
        return x_h
    
class BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BLOCK, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = FirstOctaveConv(out_channels, out_channels, kernel_size=(1,1), padding=0)
        self.conv3 = LastOctaveConv(out_channels, out_channels, kernel_size=(1,1), padding=0)
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out += residual
        out = self.relu(out)
        return out
    
class FIRST_CONV(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FIRST_CONV, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = FirstOctaveConv(out_channels, out_channels, kernel_size=(1,1), padding=0)
        self.conv3 = LastOctaveConv(out_channels, out_channels, kernel_size=(1,1), padding=0)
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out += residual
        out = self.relu(out)
        return out
    
class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.first_conv = FIRST_CONV(in_channels, 32)
        self.block1 = BLOCK(32, 64)
        self.block2 = BLOCK(64, 128)
        self.block3 = BLOCK(128, 256)
        self.block4 = BLOCK(256, 512)
        
        layerout_channel = [32,64,128,256,512]
        trans_channel_num = 32
        self.conv1x1_1 = nn.Conv2d(layerout_channel[0], trans_channel_num, 1)
        self.conv1x1_2 = nn.Conv2d(layerout_channel[1], trans_channel_num, 1)
        self.conv1x1_3 = nn.Conv2d(layerout_channel[2], trans_channel_num, 1)
        self.conv1x1_4 = nn.Conv2d(layerout_channel[3], trans_channel_num, 1)
        self.conv1x1_5 = nn.Conv2d(layerout_channel[4], trans_channel_num, 1)
        #Densely connected fusion features
        self.conv_p5 = nn.Sequential(nn.Conv2d(trans_channel_num, trans_channel_num, 1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True))
        self.conv_p4 = nn.Sequential(nn.Conv2d(trans_channel_num*2, trans_channel_num, 1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True))
        self.conv_p3 = nn.Sequential(nn.Conv2d(trans_channel_num*3, trans_channel_num, 1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True))
        self.conv_p2 = nn.Sequential(nn.Conv2d(trans_channel_num*4, trans_channel_num, 3, padding=1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True))
        self.conv_p1 = nn.Sequential(nn.Conv2d(trans_channel_num*5, trans_channel_num, 3, padding=1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True))
        
        #Attention gets weighted
        self.conv_p5_w = nn.Sequential(nn.Conv2d(trans_channel_num, trans_channel_num, 1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(trans_channel_num, 1, 1),
                                    nn.Sigmoid())
        self.conv_p4_w = nn.Sequential(nn.Conv2d(trans_channel_num, trans_channel_num, 1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(trans_channel_num, 1, 1),
                                    nn.Sigmoid())
        self.conv_p3_w = nn.Sequential(nn.Conv2d(trans_channel_num, trans_channel_num, 1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(trans_channel_num, 1, 1),
                                    nn.Sigmoid())
        self.conv_p2_w = nn.Sequential(nn.Conv2d(trans_channel_num, trans_channel_num, 3, padding=1),
                                    nn.BatchNorm2d(trans_channel_num),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(trans_channel_num, 1, 3, padding=1),
                                    nn.Sigmoid())
        
        # reduce to one dimension
        self.softmax  = nn.Softmax(dim=-1)
        self.fusion1_1x1 = nn.Sequential(
            nn.Conv2d(trans_channel_num, trans_channel_num, 3, padding=1),
                                        nn.BatchNorm2d(trans_channel_num),
                                        nn.ReLU(inplace=True))
        self.fusion2_1x1 = nn.Sequential(nn.Conv2d(trans_channel_num, trans_channel_num, 3, padding=1),
                                        nn.BatchNorm2d(trans_channel_num),
                                        nn.ReLU(inplace=True))
        self.conv_1x1 = nn.Conv2d(trans_channel_num, 1, 1)

    def forward(self, x):
        f1 = self.first_conv(x)
        f2 = self.block1(f1)
        f3 = self.block2(f2)
        f4 = self.block3(f3)
        f5 = self.block4(f4)
        
        f1 = self.conv1x1_1(f1)
        f2 = self.conv1x1_2(f2)
        f3 = self.conv1x1_3(f3)
        f4 = self.conv1x1_4(f4)
        f5 = self.conv1x1_5(f5)
        #Densely connected fusion features  
        p5 = self.conv_p5(f5)
        
        p5_to_p4 = nn.functional.interpolate(p5, f4.size()[2:])
        p4 = self.conv_p4(torch.cat([p5_to_p4, f4], dim=1))
        
        p5_to_p3 = nn.functional.interpolate(p5_to_p4, f3.size()[2:])
        p4_to_p3 = nn.functional.interpolate(p4, f3.size()[2:])
        p3 = self.conv_p3(torch.cat([p5_to_p3, p4_to_p3, f3], dim=1))
        
        p5_to_p2 = nn.functional.interpolate(p5_to_p3, f2.size()[2:])
        p4_to_p2 = nn.functional.interpolate(p4_to_p3, f2.size()[2:])
        p3_to_p2 = nn.functional.interpolate(p3, f2.size()[2:])
        p2 = self.conv_p2(torch.cat([p5_to_p2, p4_to_p2, p3_to_p2, f2], dim=1))
        
        p5_to_p1 = nn.functional.interpolate(p5_to_p2, f1.size()[2:])
        p4_to_p1 = nn.functional.interpolate(p4_to_p2, f1.size()[2:])
        p3_to_p1 = nn.functional.interpolate(p3_to_p2, f1.size()[2:])
        p2_to_p1 = nn.functional.interpolate(p2, f1.size()[2:])
        p1 = self.conv_p1(torch.cat([p5_to_p1, p4_to_p1, p3_to_p1, p2_to_p1, f1], dim=1))
    
        #Attention gets weighted
        weight5 = self.conv_p5_w(p5)
        
        b4, c4, h4, w4 = p4.size()
        weight5 = nn.functional.interpolate(weight5, (h4, w4))
        p4 = p4 * weight5 + p4
        weight4 = self.conv_p4_w(p4)
        
        b3, c3, h3, w3 = p3.size()
        weight4 = nn.functional.interpolate(weight4, (h3, w3))
        p3 = p3 * weight4 + p3
        weight3 = self.conv_p3_w(p3)
        
        b2, c2, h2, w2 = p2.size()
        weight3 = nn.functional.interpolate(weight3, (h2, w2))
        p2 = p2 * weight3 + p2
        weight2 = self.conv_p2_w(p2)
        
        b1, c1, h1, w1 = p1.size()
        weight2 = nn.functional.interpolate(weight2, (h1, w1))
        p1 = p1 * weight2 + p1
        
        #reduce to one dimension
        p5 = nn.functional.interpolate(p5, p3.size()[2:])
        p4 = nn.functional.interpolate(p4, p3.size()[2:])
        b1, c1, w1, h1 = p3.size()
        q1 = p5.view(b1, c1, -1).permute(0,2,1)
        k1 = p4.view(b1, c1, -1)
        v1 = p3.view(b1, c1, -1)
        energy1 = torch.bmm(q1, k1)
        attention1 = self.softmax(energy1)
        fusion1 = torch.bmm(v1, attention1.permute(0, 2, 1))
        fusion1 = fusion1.view(b1, c1, w1, h1)
        fusion1 = fusion1 + p3
        fusion1 = self.fusion1_1x1(fusion1)
        
        p1_ = nn.functional.interpolate(p1, fusion1.size()[2:])
        p2 = nn.functional.interpolate(p2, fusion1.size()[2:])
        b2, c2, w2, h2 = p3.size()
        q2 = p1_.view(b2, c2, -1).permute(0,2,1)
        k2 = p2.view(b2, c2, -1)
        v2 = fusion1.view(b2, c2, -1)
        energy2 = torch.bmm(q2, k2)
        attention2 = self.softmax(energy2)
        fusion2 = torch.bmm(v2, attention2.permute(0, 2, 1))
        fusion2 = fusion2.view(b2, c2, w2, h2)
        fusion2 = fusion2 + fusion1
        fusion2 = self.fusion2_1x1(fusion2)
        
        fusion = nn.functional.interpolate(fusion2, p1.size()[2:])
        out = self.conv_1x1(fusion+p1)
        return out
