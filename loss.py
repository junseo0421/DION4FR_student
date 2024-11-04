import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIM_loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_loss, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)


        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x)
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # SSIM score
        return torch.mean(1-torch.clamp((SSIM_n / SSIM_d) / 2, 0, 1))

# 24.10.15 FSP Loss
class FSP(nn.Module):
    def __init__(self):
        super(FSP, self).__init__()

    def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
        loss = F.mse_loss(self.fsp_matrix(fm_s1, fm_s2), self.fsp_matrix(fm_t1, fm_t2))

        return loss

    def fsp_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)  # B, C, H, W
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)

        return fsp