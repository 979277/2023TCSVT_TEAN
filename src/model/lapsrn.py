import torch
from torch._C import device
import torch.nn as nn
import numpy as np
import math
from IPython import embed
from model.recognizer import recognizer_builder
#from src.interfaces.base import AsterInfo
#from src.interfaces.base import TextBase
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead
import torch.nn.functional as F
from src.model.recognizer import resnet_aster
from src.model.recognizer.recognizer_builder import RecognizerBuilder
from src.interfaces.base0 import AsterInfo
import math
#import mmcv
#from interfaces.base import TextBase
#from src.interfaces.super_resolution import tpgenerator
class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        residual=x
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.sigmoid(self.conv3(x))*residual
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        '''self.avg_pool = nn.AvgPool2d(1,1)
        self.max_pool = nn.MaxPool2d(1,1)'''

        self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // 16, False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_planes // 16, in_planes, False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,_,_=x.size()
        avg_out = self.fc(self.avg_pool(x).view(b,c))
        max_out = self.fc(self.max_pool(x).view(b,c))
        out = avg_out + max_out
        att=self.sigmoid(out).view(b,c,1,1)
        return att.expand_as(x)*x

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
'''class srattention(nn.Module):
    def __init__(self,in_planes):
        super(srattention, self).__init__()
        self.conv1=nn.Conv2d(in_planes,in_planes,kernel_size=1)
        #self.avg_pool = nn.AvgPool2d(3,1,1)
        #self.max_pool = nn.MaxPool2d(3,1,1)
        self.avg_pool = nn.AdaptiveAvgPool2d((16,64))
        self.convchann0=nn.Conv2d(in_planes,in_planes//16,kernel_size=3,padding=1)
        self.convchann1=nn.Conv2d(in_planes//16,in_planes,kernel_size=3,padding=1)
        self.convspat0=nn.Conv2d(in_planes,in_planes//16,kernel_size=3,padding=1)
        self.convspat1=nn.Conv2d(in_planes//16,in_planes,kernel_size=3,padding=1)
        self.convnon=nn.Conv2d(in_planes,in_planes,kernel_size=3,padding=1)
        self.channatt=ChannelAttention(in_planes*2)
        self.channred=nn.Conv2d(2*in_planes,in_planes,kernel_size=3,padding=1)
        self.ADM = nn.Sequential(
            nn.Linear(in_planes, in_planes // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 4, 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2, bias=False)
        )
        self.avg_pool_non = nn.AdaptiveAvgPool2d(1)
        self.convlast=nn.Conv2d(in_planes,in_planes,kernel_size=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn0=nn.BatchNorm2d(in_planes)
        self.relu0=nn.ReLU()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.relu1=nn.ReLU()
        self.bn2=nn.BatchNorm2d(in_planes*2)
        self.strip_pool=SPBlock(in_planes,in_planes)
        #self.deform=mmcv.DeformConv2dFunction()
    def forward(self,x):
        a,b,c,d=x.shape
        residual=x
        x0=self.conv1(x)
        y = self.avg_pool_non(x0).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y, dim = 1)
        x_chann=self.convchann1(self.relu0((self.convchann0(self.avg_pool(x0)))))
        #x_spat=self.convspat1(self.relu1((self.convspat0(self.max_pool(x0)))))
        x_spat=self.convspat1(self.relu1((self.convspat0(self.strip_pool(x0)))))
        x_mix=torch.cat((x_chann,x_spat),1)
        x_res=self.channatt(x_mix)
        x_res=self.bn2(x_res)
        #x_res=F.interpolate(x_res,(c,d),mode='bilinear',align_corners=False)
        output=self.channred(x_res)
        output=self.bn0(output)
        x_non=self.convnon(x0)
        x_non=self.bn1(x_non)
        x1 = output * ax[:,0].view(a,1,1,1) + x_non * ax[:,1].view(a,1,1,1)
        x1 = self.lrelu(x1)
        out = self.convlast(x1)
        out = out+residual
        return out'''
'''class srattention(nn.Module):
    def __init__(self,in_planes):
        super(srattention, self).__init__()
        self.conv1=nn.Conv2d(in_planes,in_planes,kernel_size=1)
        #self.avg_pool = nn.AvgPool2d(3,1,1)
        #self.max_pool = nn.MaxPool2d(3,1,1)
        self.avg_pool = nn.AdaptiveAvgPool2d((16,64))
        self.convchann0=nn.Conv2d(in_planes,in_planes//16,kernel_size=3,padding=1)
        self.convchann1=nn.Conv2d(in_planes//16,in_planes,kernel_size=3,padding=1)
        self.convnon=nn.Conv2d(in_planes,in_planes,kernel_size=3,padding=1)
        self.channatt=ChannelAttention(in_planes*3)
        self.channred=nn.Conv2d(3*in_planes,in_planes,kernel_size=3,padding=1)
        self.ADM = nn.Sequential(
            nn.Linear(in_planes, in_planes // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 4, 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2, bias=False)
        )
        self.avg_pool_non = nn.AdaptiveAvgPool2d(1)
        self.convlast=nn.Conv2d(in_planes,in_planes,kernel_size=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn0=nn.BatchNorm2d(in_planes//16)
        self.relu0=nn.ReLU()
        self.bn1=nn.BatchNorm2d(in_planes//16)
        self.relu1=nn.ReLU()
        self.bn2=nn.BatchNorm2d(in_planes*3)
        #self.strip_pool=SPBlock(in_planes,in_planes)
        self.pool1=nn.AdaptiveAvgPool2d((None,1))
        self.conv_ver=nn.Conv2d(in_planes, in_planes//16, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_ver0=nn.Conv2d(in_planes//16, in_planes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.pool2=nn.AdaptiveAvgPool2d((1,None))
        self.conv_hor=nn.Conv2d(in_planes, in_planes//16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_hor0=nn.Conv2d(in_planes//16, in_planes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn3=nn.BatchNorm2d(in_planes)
        self.bn4=nn.BatchNorm2d(in_planes)
        #self.bn5=nn.BatchNorm2d(in_planes)
        #self.bn6=nn.BatchNorm2d(in_planes)
        self.bn7=nn.BatchNorm2d(in_planes//16)
        #self.bn8=nn.BatchNorm2d(in_planes)
        #self.bn9=nn.BatchNorm2d(in_planes)
        self.relu2=nn.ReLU()
        #self.deform=mmcv.DeformConv2dFunction()
    def forward(self,x):
        a,b,h,w=x.shape
        residual=x
        x0=self.conv1(x)
        y = self.avg_pool_non(x0).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y, dim = 1)
        #_, _, h, w = x.size()
        #x1 = self.pool1(x0)
        x1 = self.conv_ver(x0)
        x1 =self.bn0(x1)
        x1 = self.relu0(x1)
        x1 = self.conv_ver0(x1)
        #x1 =self.bn5(x1)
        
        #x1 = x1.expand(-1, -1, h, w)
        
        #x2 = self.pool2(x0)
        x2 = self.conv_hor(x0)
        x2 =self.bn1(x2)
        x2 = self.relu1(x2)
        x2 =self.conv_hor0(x2)
        #x2 =self.bn6(x2)
        
        x3=self.convchann0(x0)
        x3=self.bn7(x3)
        x3=self.relu2(x3)
        x3=self.convchann1(x3)
        #x3=self.bn8(x3)
        #x2 = x2.expand(-1, -1, h, w)
        x_mix=torch.cat((x1,x2,x3),1)
        x_res=self.channatt(x_mix)
        x_res=self.bn2(x_res)
        #x_res=F.interpolate(x_res,(c,d),mode='bilinear',align_corners=False)
        output=self.channred(x_res)
        output=self.bn3(output)
    
        #x_non=self.avg_pool(x0)
        x_non=self.convnon(x0)
        x_non=self.bn4(x_non)
        x3 = output * ax[:,0].view(a,1,1,1) + x_non * ax[:,1].view(a,1,1,1)
        x3 = self.lrelu(x3)
        out = self.convlast(x3)
        #out =self.bn9(out)
        out = out+residual
        return out
'''
class srattention(nn.Module):  #OCAM
    def __init__(self,in_planes):
        super(srattention, self).__init__()
        self.conv1=nn.Conv2d(in_planes,in_planes,kernel_size=1)
        #self.avg_pool = nn.AvgPool2d(3,1,1)
        #self.max_pool = nn.MaxPool2d(3,1,1)
        #self.avg_pool = nn.AdaptiveAvgPool2d((16,64))
        self.convchann0=nn.Conv2d(in_planes,in_planes//16,kernel_size=3,padding=1)
        self.convchann1=nn.Conv2d(in_planes//16,in_planes,kernel_size=3,padding=1)
        self.convnon=nn.Conv2d(in_planes,in_planes,kernel_size=3,padding=1)
        self.channatt=ChannelAttention(in_planes*3)
        self.channred=nn.Conv2d(3*in_planes,in_planes,kernel_size=3,padding=1)
        self.ADM = nn.Sequential(
            nn.Linear(in_planes, in_planes // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 4, 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2, bias=False)
        )
        self.avg_pool_non = nn.AdaptiveAvgPool2d(1)
        self.convlast=nn.Conv2d(in_planes,in_planes,kernel_size=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn0=nn.BatchNorm2d(in_planes//16)
        self.relu0=nn.ReLU()
        self.bn1=nn.BatchNorm2d(in_planes//16)
        self.relu1=nn.ReLU()
        self.bn2=nn.BatchNorm2d(in_planes*3)
        #self.strip_pool=SPBlock(in_planes,in_planes)
        #self.pool1=nn.AdaptiveAvgPool2d((None,1))
        self.conv_ver=nn.Conv2d(in_planes, in_planes//16, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_ver0=nn.Conv2d(in_planes//16, in_planes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        #self.pool2=nn.AdaptiveAvgPool2d((1,None))
        self.conv_hor=nn.Conv2d(in_planes, in_planes//16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_hor0=nn.Conv2d(in_planes//16, in_planes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn3=nn.BatchNorm2d(in_planes)
        self.bn4=nn.BatchNorm2d(in_planes)
        '''self.bn5=nn.BatchNorm2d(in_planes)
        self.bn6=nn.BatchNorm2d(in_planes)'''
        self.bn7=nn.BatchNorm2d(in_planes//16)
        #self.bn8=nn.BatchNorm2d(in_planes)
        #self.pool3=nn.AvgPool2d(3,1,1)
        #self.bn9=nn.BatchNorm2d(in_planes)
        self.relu2=nn.ReLU()
        #self.deform=mmcv.DeformConv2dFunction()
    def forward(self,x):
        a,b,h,w=x.shape
        residual=x
        x0=self.conv1(x)
        y = self.avg_pool_non(x0).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y, dim = 1)
        #_, _, h, w = x.size()
        #x1 = self.pool1(x0)
        x1 = self.conv_ver(x0)
        x1 =self.bn0(x1)
        x1 = self.relu0(x1)
        x1 = self.conv_ver0(x1)
        '''x1 =self.bn5(x1)
        x1 = self.pool1(x1)
        
        x1 = x1.expand(-1, -1, h, w)'''
        
        #x2 = self.pool2(x0)
        x2 = self.conv_hor(x0)
        x2 =self.bn1(x2)
        x2 = self.relu1(x2)
        x2 =self.conv_hor0(x2)
        '''x2 =self.bn6(x2)
        x2 = self.pool2(x2)
        x2 = x2.expand(-1, -1, h, w)'''
        x3=self.convchann0(x0)
        x3=self.bn7(x3)
        x3=self.relu2(x3)
        x3=self.convchann1(x3)
        #x3=self.bn8(x3)
        #x3=self.pool3(x3)
        x_mix=torch.cat((x1,x2,x3),1)
        x_res=self.channatt(x_mix)
        x_res=self.bn2(x_res)
        output=self.channred(x_res)
        output=self.bn3(output)
    
        #x_non=self.avg_pool(x0)
        x_non=self.convnon(x0)
        x_non=self.bn4(x_non)
        x4 = output * ax[:,0].view(a,1,1,1) + x_non * ax[:,1].view(a,1,1,1)
        x4 = self.lrelu(x4)
        out = self.convlast(x4)
        #out =self.bn9(out)
        out = out+residual
        return out
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True
    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x
class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)
    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()#view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x
class segattention(nn.Module):  #tsca
    def __init__(self):
        super(segattention,self).__init__()
        self.convatt0=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.convatt1=nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.convatt2=nn.Conv2d(16,1,kernel_size=1)
        self.convatt3=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        att0=self.convatt0(x)
        att1=self.convatt1(att0)
        att2=self.convatt2(att1)
        att_inf=self.sigmoid(att2)
        att3=self.convatt3(x)
        att3=att3*math.e**(att_inf)
        att4=att3+att0
        return att4,att_inf
class RecurrentResidualBlock(nn.Module): #teab-second-level
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)
        self.ca=ChannelAttention(channels)
        self.sa=SpatialAttention(7)
        self.projection = nn.Conv2d(channels+32, channels, kernel_size=1, padding=0)
        self.sratt=srattention(channels)
        #self.FS=iAFF()
       

    def forward(self, x):#,tpf
        
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        #residual = self.conv2(residual)
        #x0=residual
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        #x1=residual
        '''residual = self.ca(residual)*residual
        residual=self.sa(residual)*residual'''
        residual = self.gru1((residual).transpose(-1, -2)).transpose(-1, -2)
        
        #t=residual
        
        
        '''residual=self.cdcl(residual)
        residual=self.cdclr(residual)
        residual=self.maxpooll(residual)
        ll=residual
        residual=self.cdch(residual)
        residual=self.cdchr(residual)
        residual=self.cdcm(residual)
        residual=self.cdcmr(residual)
residual=self.maxpooll(residua
        ml=residual
        residual=self.cdc(residual)
        residual=self.cdch(residual)
        residual=self.cdchr(residual)
        residual=self.maxpooll(residual)
        hl=residual
        ll=self.sal(ll)*ll
        ml=self.sam(ml)*ml
        hl=self.sah(hl)*hl
        #residual=self.a2b(residual)
        #residual=t+residual
        ll=F.interpolate(ll,[x.size(2),x.size(3)])
        hl=F.interpolate(hl,[x.size(2),x.size(3)])
        ml=F.interpolate(ml,[x.size(2),x.size(3)])
        
        residual=torch.cat([ll,ml,hl],dim=1)
        residual=F.interpolate(residual,[x.size(2),x.size(3)])
        #print(residual.size())
        
        residual=self.cdcrr(residual)'''
        
        #residual=self.gru2(x+residual)
        residual=self.gru2(residual+x)
        
        '''tpf = torch.nn.functional.interpolate(tpf, (residual.shape[2], residual.shape[3]), mode='bicubic', align_corners=True)
        res1 = self.projection(torch.cat([residual, tpf], dim=1))
        residual=residual+res1'''
        '''residual = self.ca(residual)*residual
        t=residual
        residual=self.sa(residual)*t'''
        residual=self.sratt(residual)
        return residual

'''def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()'''


'''class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output
'''
class RecurrentResidualBlock0(nn.Module): #teab_first-level
    def __init__(self, channels):
        super(RecurrentResidualBlock0, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)
        self.ca=ChannelAttention(channels)
        self.sa=SpatialAttention(7)
        self.projection = nn.Conv2d(channels+32, channels, kernel_size=1, padding=0)
        #self.sratt=srattention(channels)
        #self.segattention=segattention()
        #self.FS=iAFF()
       

    def forward(self, x):#,tpf
        
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        #residual = self.conv2(residual)
        #x0=residual
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        #x1=residual
        '''residual = self.ca(residual)*residual
        residual=self.sa(residual)*residual'''
        residual = self.gru1((residual).transpose(-1, -2)).transpose(-1, -2)
        
        #t=residual
        
        
        
        #residual=self.gru2(x+residual)
        residual=self.gru2(residual+x)
        
        '''tpf = torch.nn.functional.interpolate(tpf, (residual.shape[2], residual.shape[3]), mode='bicubic', align_corners=True)
        res1 = self.projection(torch.cat([residual, tpf], dim=1))
        residual=residual+res1'''
        #residual=self.sratt(residual)
        residual = self.ca(residual)
        t=residual
        residual=self.sa(residual)*t
        #residual,att_inf=self.segattention(residual)
        return residual
class LapSRN(nn.Module):#nn.module   TEAN
    def __init__(self, scale_factor=4, in_planes=4, STN=False, width=128, height=32):
        super(LapSRN, self).__init__()
        
        self.scale_factor = scale_factor
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=9, padding=4),#卷积提取浅层特征
            nn.PReLU()
            # nn.ReLU()
        )
        
        '''for i in range(5):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(64))#设置循环残差块，5个，block2-6'''
        for i in range(5):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(64))#设置循环残差块，5个，block2-6
        setattr(self, 'block%d' % (7),#设置block7
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64)
                ))
        block_ = [UpsampleBLock(64, 2) for _ in range(1)]#设置一个上采样块
        #block_ =[nn.Conv2d(64, in_planes, kernel_size=9, padding=4)]
        block_.append(nn.Conv2d(64, in_planes, kernel_size=9, padding=4))#卷积层'''
        
        setattr(self, 'block%d' % (8), nn.Sequential(*block_))#上述两者合成一个块，block8
        self.block9 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=9, padding=4),#卷积提取浅层特征
            nn.PReLU()
            # nn.ReLU()
        )
        '''for i in range(5):
            setattr(self, 'block%d' % (i + 10), RecurrentResidualBlock(64))#设置循环残差块，5个，block10-14'''
        for i in range(5):
            setattr(self, 'block%d' % (i + 10), RecurrentResidualBlock0(64))#设置循环残差块，4个，block10-14
        setattr(self, 'block%d' % (15),#设置block15
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64)
                ))
        block1_ = [UpsampleBLock(64, 2) for _ in range(1)]#设置一个上采样块
        block1_.append(nn.Conv2d(64, in_planes, kernel_size=9, padding=4))#卷积层
        #block1_ =[nn.Conv2d(64, in_planes, kernel_size=9, padding=4)]
        setattr(self, 'block%d' % (16), nn.Sequential(*block1_))#上述两者合成一个块，block16
        '''self.block17 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=9, padding=4),#卷积提取浅层特征
            nn.PReLU()
            # nn.ReLU()
        )
        for i in range(5):
            setattr(self, 'block%d' % (i + 18), RecurrentResidualBlock0(64))#设置循环残差块，5个，block18-22
        setattr(self, 'block%d' % (23),#设置block23
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64)
                ))
        block2_ = [UpsampleBLock(64, 2) for _ in range(1)]#设置一个上采样块
        block2_.append(nn.Conv2d(64, in_planes, kernel_size=9, padding=4))#卷积层
        setattr(self, 'block%d' % (24), nn.Sequential(*block2_))#上述两者合成一个块'''
        self.tps_inputsize = [32, 64]
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=4,
                num_ctrlpoints=num_control_points,
                activation='none')
        self.upb=UpsampleBLock(64,2)
        self.ca=ChannelAttention(64)
        self.sa=SpatialAttention(7)
        '''self.transf = nn.ModuleList()
        self.transf.append(nn.ConvTranspose2d(16, 16, 3, stride=2))
        self.transf.append(nn.BatchNorm2d(16))
        self.transf.append(nn.PReLU())
        self.transf.append(nn.ConvTranspose2d(16, 16, 3, stride=(2, 1)))
        self.transf.append(nn.BatchNorm2d(16))
        self.transf.append(nn.PReLU())
        self.transf.append(nn.ConvTranspose2d(16, 32, 3, stride=(2, 1)))
        self.transf.append(nn.BatchNorm2d(32))
        self.transf.append(nn.PReLU())
        self.transf1 = nn.ModuleList()
        self.transf1.append(nn.ConvTranspose2d(32, 32, 3, stride=2))
        self.transf1.append(nn.BatchNorm2d(32))
        self.transf1.append(nn.PReLU())
        self.transf1.append(nn.ConvTranspose2d(32, 32, 3, stride=(2, 1)))
        self.transf1.append(nn.BatchNorm2d(32))
        self.transf1.append(nn.PReLU())
        self.transf1.append(nn.ConvTranspose2d(32, 32, 3, stride=(2, 1)))
        self.transf1.append(nn.BatchNorm2d(32))
        self.transf1.append(nn.PReLU())'''
        
        self.upb0=UpsampleBLock(4,2)
        #self.upb1=UpsampleBLock(4,2)
        '''self.aster_info = AsterInfo('all')
        self.astertp0=RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=self.aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=self.aster_info.max_len,
                                             eos=self.aster_info.char2id[self.aster_info.EOS], STN_ON=True)
        self.astertp0.load_state_dict(torch.load('/home/tongji/shurui/TextZoom-master/src/pth/demo.pth.tar')['state_dict'])
        self.astertp=self.astertp0.encoder'''
        self.loss1=nn.L1Loss()
        self.loss2=nn.KLDivLoss()
        self.convw=nn.Conv2d(8,4,kernel_size=3,padding=1)
        self.convrec=nn.Conv2d(4,4,kernel_size=1)
        self.sigmoid=nn.Sigmoid()
        self.segattention=segattention()
        self.sratt=srattention(64)
        self.convw_img=nn.Conv2d(2,1,kernel_size=3,padding=1)
        self.convrec_img=nn.Conv2d(1,1,kernel_size=1)
        #self.sratt0=srattention(64)
        '''self.block9 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=9, padding=4),#卷积提取浅层特征
            nn.PReLU()
            # nn.ReLU()
        )'''
        #self.sratt0=srattention(64)
    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)
    def forward(self, x):
        att_inf_list=[]
        if self.stn and self.training:
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        input0=2*x[:,:3,:,:]-1
        '''tp=self.astertp(input0)
        tpf = tp.contiguous().view(x.shape[0], 16, 1, -1)
        for module in self.transf:
            tpf = module(tpf) #32通道'''
        block = {'1': self.block1(x)}
        for i in range(6):#block2-5
            if i<5:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])#,tpf
            else :
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])
        block['7']=self.sratt(block['7'])
        block[str(8)] = getattr(self, 'block%d' % (8)) \
            (block[str(7)])
        feat0=torch.cat((block['8'],self.upb0(x)),1)
        sw0=self.sigmoid(self.convrec(self.convw(feat0)))#fusion权重
        HR_2x = torch.tanh((1-sw0)*(self.upb0(x))+sw0*block['8'])
        
        '''input1=2*HR_2x[:,:3,:,:]-1
        tp1=self.astertp(input1)
        tpf1 = tp1.contiguous().view(x.shape[0], 32, 1, -1)
        for module in self.transf1:
            tpf1 = module(tpf1)#32通道 '''
        block['9']=self.block9(HR_2x)
        for i in range(6):#block10-13
            if i<5:
                block[str(i + 10)]= getattr(self, 'block%d' % (i + 10))(block[str(i + 9)])#,tpf1
                block[str(i+10)],att_inf32=self.segattention(block[str(i+10)])
                att_inf_list.append(att_inf32)
            else :
                block[str(i + 10)] = getattr(self, 'block%d' % (i + 10))(block[str(i + 9)]) 
        block['15']=self.ca(block['15'])
        t=block['15']
        block['15']=(self.sa(block['15']))*t
        
        block[str(16)] = getattr(self, 'block%d' % (16)) \
        (block[str(15)])

        feat1=torch.cat((block['16'],self.upb0(HR_2x)),1)
        sw1=self.sigmoid(self.convrec(self.convw(feat1)))#fusion权重
        HR_4x=torch.tanh((1-sw1)*self.upb0(HR_2x)+sw1*block['16'])
        '''HR_4x0=F.interpolate(HR_4x,(32,128),mode='bicubic',align_corners=True)
        input2=2*HR_4x0[:,:3,:,:]-1
        #input2=input2.contiguous().view(x.shape[0], 3, -1)
        #import ipdb;ipdb.set_trace()
        tp2=self.astertp(input2)
        tpf2 = tp2.contiguous().view(x.shape[0], 32, 1, -1)
        for module in self.transf1:
            tpf2 = module(tpf2)#32通道 
        block['17']=self.block17(HR_4x)
        for i in range(6):#block18-23
            if i<5:
                block[str(i + 18)]= getattr(self, 'block%d' % (i + 18))(block[str(i + 17)],tpf2)#,att_inf_image32
                block[str(i+18)],att_inf32=self.segattention(block[str(i+18)])
                att_inf_list.append(att_inf32)
            else :
                block[str(i + 18)] = getattr(self, 'block%d' % (i + 18))(block[str(i + 17)]) 
        block['23']=self.ca(block['23'])
        t=block['23']
        block['23']=(self.sa(block['23']))*t
        
        block[str(24)] = getattr(self, 'block%d' % (24)) \
        (block[str(23)])

        feat2=torch.cat((block['24'],self.upb0(HR_4x)),1)
        sw2=self.sigmoid(self.convrec(self.convw(feat2)))#fusion权重
        HR_8x=torch.tanh((1-sw2)*self.upb0(HR_4x)+sw2*block['24'])'''
        
        att_inf_sum=torch.zeros(HR_2x.size(0),1,HR_2x.size(2),HR_2x.size(3)).cuda()

        for j in range(0,len(att_inf_list)):
            att_inf_sum=att_inf_sum+att_inf_list[j]
        att_inf_avg=att_inf_sum/len(att_inf_list)


        return HR_2x,HR_4x,att_inf_avg