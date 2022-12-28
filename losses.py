from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from math import exp
import config
import torch

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
    def forward(self, x ):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return torch.mean(k)

class dice(nn.Module):
  def __init__(self):
    super(dice, self).__init__()

  def forward(self, input, target, smooth=1.):
    inputs = input.view(-1)
    targets = target.view(-1)
    intersection = (inputs * targets).sum()
    dice =  (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

        self.dice = dice()

    def forward(self, inputs, targets, smooth=1.):
             
        dice = self.dice(inputs, targets, smooth)                       
        dice_loss = 1 - dice  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # Dice_BCE = BCE + dice_loss
        
        return dice_loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
        self.gray_l1 = nn.L1Loss()
        self.tm_l1 = nn.L1Loss()
        self.atmos_l1 = nn.L1Loss()
        self.out_l1 = nn.L1Loss()
        # self.out_ssim = SSIM()
        # self.l_color = L_color()
        # self.l_tv = L_TV()
        
    def forward(self, orig_normal, orig_gray, orig_tm, orig_atmos, pred_normal, pred_gray, pred_tm, pred_atmos):
        gray_l1 = self.gray_l1(orig_gray, pred_gray)
        tm_l1 = self.tm_l1(orig_tm, pred_tm)
        atmos_l1 = self.atmos_l1(orig_atmos, pred_atmos)
        out_l1 = self.out_l1(orig_normal, pred_normal)
        # out_ssim = (1 - self.out_ssim(orig_normal, pred_normal))
        # out_color = self.l_color(pred_normal)
        # out_tv = self.l_tv(pred_normal)
        loss = gray_l1 + tm_l1 + atmos_l1 + out_l1 #+ out_ssim + out_color + out_tv
        return loss, tm_l1, atmos_l1, out_l1#, out_ssim, out_color, out_tv

class CoarseLoss(nn.Module):
    def __init__(self):
        super(CoarseLoss, self).__init__()
        
        self.gray_l1 = nn.L1Loss()
        self.gray_ssim = SSIM()
        self.tm_l1 = nn.L1Loss()
        self.atmos_l1 = nn.L1Loss()
        
    def forward(self, orig_gray, orig_tm, orig_atmos, pred_gray, pred_tm, pred_atmos):
        gray_l1 = self.gray_l1(orig_gray, pred_gray)
        gray_ssim = (1.0 - self.gray_ssim(orig_gray, pred_gray))
        tm_l1 = self.tm_l1(orig_tm, pred_tm)
        atmos_l1 = self.atmos_l1(orig_atmos, pred_atmos)
        loss = gray_l1*0.25 + gray_ssim*0.25 + tm_l1*2 + atmos_l1
        return loss, gray_l1, gray_ssim, tm_l1, atmos_l1

class FineLoss(nn.Module):
    def __init__(self):
        super(FineLoss, self).__init__()
        self.out_l1 = nn.L1Loss()
        self.l_tv = L_TV()
    
    def forward(self, orig_normal, pred_normal):
        out_l1 = self.out_l1(orig_normal, pred_normal)
        l_tv = self.l_tv(pred_normal)
        loss = out_l1 + l_tv
        return loss, out_l1, l_tv
