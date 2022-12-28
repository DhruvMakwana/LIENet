import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils import getFinalImage

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)
        
    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)

class g_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(g_net, self).__init__()

        base_channel = 16
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 1, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
        
    def forward(self, x):
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray
    
class tm_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(tm_net, self).__init__()

        base_channel = 16
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 1, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
        
    def forward(self, x):
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray
    
class atmos_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(atmos_net, self).__init__()

        base_channel = 16
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
        
    def forward(self, x):
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))
        x = x.view(x.shape[0], -1)
        gray = (torch.tanh(x) + 1) / 2
        return gray

class refine_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(refine_net, self).__init__()

        base_channel = 16
        
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel*2, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Down_scale(base_channel*2),
            BasicConv(base_channel*4, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Down_scale(base_channel*4),
        ])
        
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel*8),
            BasicConv(base_channel*8, base_channel*4, 3, 1),
            nn.Sequential(*[RB(base_channel*4) for _ in range(depth[2])]),
            Up_scale(base_channel*4),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[RB(base_channel*2) for _ in range(depth[1])]),
            Up_scale(base_channel*2),
            BasicConv(base_channel*2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(7, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x
        
    def forward(self, x):
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray

class LIENet(nn.Module):
    def __init__(self):
        super(LIENet, self).__init__()
        self.g_net = g_net()
        self.tm_net = tm_net()
        self.atmos_net = atmos_net()
        self.refine_net = refine_net()

    def forward(self, x):
        gray = self.g_net(x)
        tm = self.tm_net(x)
        atmos = self.atmos_net(x)
        coarsemap = getFinalImage(x, atmos, tm)
        out = self.refine_net(torch.cat([x, gray, coarsemap], 1))
        return gray, tm, atmos, coarsemap, out

class Discriminator(nn.Module):
    def __init__(self, channels_img=5, features_d=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, features_d*4, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

def test():
    model = LIENet().to("cuda")
    x = torch.randn((1, 3, 512, 512)).to("cuda")
    gray_pred, tm_pred, atmos_pred, refined_map, out_pred = model(x)
    print(summary(model, (3, 512, 512)))
    print(gray_pred.shape, tm_pred.shape, atmos_pred.shape, refined_map.shape, out_pred.shape)

if __name__ == "__main__":
    test()
