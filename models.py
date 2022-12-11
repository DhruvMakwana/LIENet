import torch
import torch.nn as nn
from torchvision import *
import torch.nn.functional as F

class backbone(nn.Module):
  def __init__(self):
    super(backbone, self).__init__()

    model = models.efficientnet_b6(pretrained=True)
    m1 = list(model.features.children())
    self.l1 = nn.Sequential(*list(model.features.children())[:2])
    self.l2 = nn.Sequential(*list(model.features[2][:3]))
    self.l3 = nn.Sequential(*list(model.features[3][:4]))
    self.l4 = nn.Sequential(*list(model.features[4][:4]))
    self.l5 = nn.Sequential(*list(model.features[5][:2]))
    self.l6 = nn.Sequential(*list(model.features[6][:2]))

  def forward(self, x):
    x1 = self.l1(x)
    x2 = self.l2(x1)
    x3 = self.l3(x2)
    x4 = self.l4(x3)
    x5 = self.l5(x4)
    x6 = self.l6(x5)
    return [x1, x2, x3, x5, x6]


class ConvbnRelu(nn.Module):
  def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1):
    super(ConvbnRelu, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(outchannels),
        nn.LeakyReLU(0.2),
    )
  def forward(self, x):
    return self.conv(x)

class SEAttention(nn.Module):   #it gives channel attention
    def __init__(self, in_channels, reduced_dim=16):  #input_shape ---> output_shape
        super(SEAttention, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.c1 = ConvbnRelu(inchannels=in_c, outchannels=out_c)
        self.c2 = ConvbnRelu(inchannels=out_c, outchannels=out_c)
        self.c3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SEAttention(in_channels=out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)
        x4 = x2 + x3
        x4 = self.relu(x4)
        return x4

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.r1(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(skip_c+out_c, out_c)
        # self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        # x = self.r2(x)
        return x

class GBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(GBlock, self).__init__()

        self.c = nn.Conv2d(in_c+in_c, out_c, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2], axis=1)
        x = self.c(x)
        x = self.sig(x)
        x = x * x3
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Refinement(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[16, 32, 64, 128],
    ):
        super(Refinement, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class LIE_Prior(nn.Module):
  def __init__(self):
    super(LIE_Prior, self).__init__()
    
    self.backbone = backbone()
    self.refine = Refinement(in_channels=4, out_channels=3)

    self.ld1 = DecoderBlock(in_c=344, skip_c=200, out_c=256)
    self.hd1 = DecoderBlock(in_c=344, skip_c=200, out_c=256)
    self.id1 = DecoderBlock(in_c=344, skip_c=200, out_c=256)
    self.g1 = GBlock(in_c=256, out_c=256)

    self.ld2 = DecoderBlock(in_c=256, skip_c=72, out_c=128)
    self.hd2 = DecoderBlock(in_c=256, skip_c=72, out_c=128)
    self.id2 = DecoderBlock(in_c=256, skip_c=72, out_c=128)
    self.g2 = GBlock(in_c=128, out_c=128)

    self.ld3 = DecoderBlock(in_c=128, skip_c=40, out_c=64)
    self.hd3 = DecoderBlock(in_c=128, skip_c=40, out_c=64)
    self.id3 = DecoderBlock(in_c=128, skip_c=40, out_c=64)
    self.g3 = GBlock(in_c=64, out_c=64)

    self.ld4 = DecoderBlock(in_c=64, skip_c=32, out_c=32)
    self.hd4 = DecoderBlock(in_c=64, skip_c=32, out_c=32)
    self.id4 = DecoderBlock(in_c=64, skip_c=32, out_c=32)
    self.g4 = GBlock(in_c=32, out_c=32)

    self.ld5 =  nn.Sequential(  nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                                nn.Conv2d(32, 1, 3, 1, 1),
                                nn.Sigmoid()
                            )#ResidualBlock(in_c=32, out_c=3, last=True)
    self.hd5 = nn.Sequential(   nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                                nn.Conv2d(32, 1, 3, 1, 1),
                                nn.Sigmoid()
                            )#ResidualBlock(in_c=32, out_c=1, last=True)
    self.id5 = nn.Sequential(   nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                                nn.Conv2d(32, 3, 3, 1, 1),
                                nn.Sigmoid()
                            )#ResidualBlock(in_c=32, out_c=3, last=True)
    
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):

    x1, x2, x3, x4, x5 = self.backbone(x)

    """block 1"""
    l1 = self.ld1(x5,x4)  
    h1 = self.hd1(x5,x4)  
    i1 = self.id1(x5,x4) 
    i1 = self.g1(l1,h1,i1)

    """block 2"""
    l2 = self.ld2(l1,x3)  
    h2 = self.hd2(h1,x3)  
    i2 = self.id2(i1,x3) 
    i2 = self.g2(l2,h2,i2)

    """block 3"""
    l3 = self.ld3(l2,x2)  
    h3 = self.hd3(h2,x2)  
    i3 = self.id3(i2,x2) 
    i3 = self.g3(l3,h3,i3)

    """block 4"""
    l4 = self.ld4(l3,x1)  
    h4 = self.hd4(h3,x1)  
    i4 = self.id4(i3,x1) 
    i4 = self.g4(l4,h4,i4)

    """block 5 [last block]"""
    l5 = self.ld5(l4)
    h5 = self.hd5(h4)
    i5 = self.id5(i4)
    i5 = F.avg_pool2d(i5, (i5.shape[2], i5.shape[3]))

    out = (x - ((1.0 - h5) * i5)) / (h5)
    out = self.sigmoid(out)
    out = self.refine(torch.cat([l5, out], 1))

    return l5,h5,i5,out

def test():
    from torchsummary import summary
    x = torch.randn((1, 3, 512, 512)).to("cuda")
    model = LIE_Prior().to("cuda")
    gray_pred, tm_pred, atmos_pred, out_pred = model(x)
    print(summary(model, (3, 512, 512)))
    print(gray_pred.shape, tm_pred.shape, atmos_pred.shape, out_pred.shape)

if __name__ == "__main__":
    test()