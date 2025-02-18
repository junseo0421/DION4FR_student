# from pytorch_wavelets import DWTForward, DWTInverse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from pytorch_wavelets import DWTForward, DWTInverse

class SSIM_loss(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_loss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
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

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # SSIM score
        return torch.mean(1-torch.clamp((SSIM_n / SSIM_d) / 2, 0, 1))


class FSP(nn.Module):
    def __init__(self):
        super(FSP, self).__init__()

    def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
        loss = F.mse_loss(self.fsp_matrix(fm_s1, fm_s2), self.fsp_matrix(fm_t1, fm_t2))

        return loss

    def fsp_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)

        return fsp


class AT(nn.Module):
    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)

        return am


class OFD(nn.Module):
    def __init__(self):
        super(OFD, self).__init__()

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)

        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t)**2 * mask)

        return loss

    def get_margin(self, fm, eps=1e-6):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask

        margin = masked_fm.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True)+eps)

        return margin


class OFD_CON(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OFD_CON, self).__init__()
        self.connector = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        fm_s = self.connector(fm_s)

        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t)**2 * mask)

        return loss

    def get_margin(self, fm, eps=1e-6):
        mask = (fm < 0.0).float()
        masked_fm = fm * mask

        margin = masked_fm.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True)+eps)

        return margin


class FAKD(nn.Module):
    def __init__(self):
        super(FAKD, self).__init__()

    def forward(self, fm_s, fm_t):
        loss = F.l1_loss(self.spatial_similarity(fm_s), self.spatial_similarity(fm_t))

        return loss

    def spatial_similarity(self, fm):
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
        s = norm_fm.transpose(1, 2).bmm(norm_fm)
        s = s.unsqueeze(1)
        return s


class FFT_Module(nn.Module):
    def __init__(self):
        super(FFT_Module, self).__init__()

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.fft_complex(fm_s), self.fft_complex(fm_t))
        return loss

    def fft_complex(self, fm, eps=1e-6):
        fm_fft = torch.fft.fft2(fm, norm="ortho")

        return fm_fft


class RKD(nn.Module):
    def __init__(self, w_dist=1, w_angle=2):
        super(RKD, self).__init__()

        self.w_dist = w_dist
        self.w_angle = w_angle

    def forward(self, feat_s, feat_t):
        loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
               self.w_angle * self.rkd_angle(feat_s, feat_t)

        return loss

    def rkd_dist(self, feat_s, feat_t):
        feat_t_dist = self.pdist(feat_t, squared=False)
        mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
        feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
        norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
        feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

        feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class SASA_Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=7, num_heads=8, image_size=224, inference=False):
        super(SASA_Layer, self).__init__()
        self.kernel_size = min(kernel_size, image_size)  # receptive field shouldn't be larger than input H/W
        self.num_heads = num_heads
        self.dk = self.dv = in_channels
        self.dkh = self.dk // self.num_heads
        self.dvh = self.dv // self.num_heads

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dk % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"

        self.k_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1).to(device)
        self.q_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1).to(device)
        self.v_conv = nn.Conv2d(self.dv, self.dv, kernel_size=1).to(device)

        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn(self.dk // 2, self.kernel_size, 1), requires_grad=True)
        self.rel_encoding_w = nn.Parameter(torch.randn(self.dk // 2, 1, self.kernel_size), requires_grad=True)

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Compute k, q, v
        padded_x = F.pad(x, [(self.kernel_size - 1) // 2, (self.kernel_size - 1) - ((self.kernel_size - 1) // 2),
                             (self.kernel_size - 1) // 2, (self.kernel_size - 1) - ((self.kernel_size - 1) // 2)])
        k = self.k_conv(padded_x)
        q = self.q_conv(x)
        v = self.v_conv(padded_x)

        # Unfold patches into [BS, num_heads*depth, horizontal_patches, vertical_patches, kernel_size, kernel_size]
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        # Reshape into [BS, num_heads, horizontal_patches, vertical_patches, depth_per_head, kernel_size*kernel_size]
        k = k.reshape(batch_size, self.num_heads, height, width, self.dkh, -1)
        v = v.reshape(batch_size, self.num_heads, height, width, self.dvh, -1)

        # Reshape into [BS, num_heads, height, width, depth_per_head, 1]
        q = q.reshape(batch_size, self.num_heads, height, width, self.dkh, 1)

        qk = torch.matmul(q.transpose(4, 5), k)
        qk = qk.reshape(batch_size, self.num_heads, height, width, self.kernel_size, self.kernel_size)

        # Add positional encoding
        qr_h = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_h)
        qr_w = torch.einsum('bhxydz,cij->bhxyij', q, self.rel_encoding_w)
        qk += qr_h
        qk += qr_w

        qk = qk.reshape(batch_size, self.num_heads, height, width, 1, self.kernel_size * self.kernel_size)
        weights = F.softmax(qk, dim=-1)

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn_out = torch.matmul(weights, v.transpose(4, 5))
        attn_out = attn_out.reshape(batch_size, -1, height, width)
        return attn_out


class FAM_Module(nn.Module):
    def __init__(self, in_channels, out_channels, shapes):
        super(FAM_Module, self).__init__()

        """
        feat_s_shape, feat_t_shape
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes
        #  print(self.shapes)
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        # self.out_channels = feat_t_shape[1]
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes, dtype=torch.cfloat))
        self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1)

        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        if isinstance(x, tuple):
            x, cuton = x
        else:
            cuton = 0.1
        batchsize = x.shape[0]
        x_ft = torch.fft.fft2(x, norm="ortho")
        #  print(x_ft.shape)
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        batch_fftshift = batch_fftshift2d(out_ft)

        # do the filter in here
        h, w = batch_fftshift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
        # the value of center pixel is zero.
        batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0
        # test with batch shift
        out_ft = batch_ifftshift2d(batch_fftshift)
        out_ft = torch.view_as_complex(out_ft)
        # Return to physical space
        out = torch.fft.ifft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho").real
        out2 = self.w0(x)
        return self.rate1 * out + self.rate2 * out2

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

def batch_fftshift2d(x):
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)

    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)

    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None)
                  if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None)
                  if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap


class CWD(nn.Module):
    def __init__(self, norm_type='channel', divergence='kl', temperature=4.0):
        super(CWD, self).__init__()

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self, preds_S, preds_T):
        n, c, h, w = preds_S.shape
        # import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)

        # item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        # import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature ** 2)


class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s  = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t  = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss


class SRD(nn.Module):
    """
    Args:
        s_dim: the dimension of student's feature
        t_dim: the dimension of teacher's feature
    """
    def __init__(self, s_dim, t_dim):
        super(SRD, self).__init__()
        self.s_dim = s_dim
        self.t_dim = t_dim
        # self.alpha = alpha

        self.embed = nn.Linear(s_dim, t_dim)
        self.bn_s = torch.nn.BatchNorm1d(t_dim, eps=1e-5, affine=False)
        self.bn_t = torch.nn.BatchNorm1d(t_dim, eps=1e-5, affine=False)

    # def forward_simple(self, z_s, z_t):
    #     f_s = z_s
    #     f_t = z_t
    #
    #     # must reshape the transformer repr
    #     b = f_s.shape[0]
    #     f_s = f_s.transpose(1, 2).view(b, -1, 14, 14)
    #     f_s = self.embed(f_s)
    #
    #     f_s = F.normalize(f_s, dim=1)
    #     f_t = F.normalize(f_t, dim=1)
    #
    #     return F.mse_loss(f_s, f_t)

    def forward(self, z_s, z_t):
        b, c, h, w = z_s.shape
        b1, c1, h1, w1 = z_t.shape
        z_s = z_s.view(b, c, -1).mean(2)
        z_t = z_t.view(b1, c1, -1).mean(2)

        f_s = z_s
        f_t = z_t

        f_s = self.embed(f_s)
        n, d = f_s.shape

        f_s_norm = self.bn_s(f_s)
        f_t_norm = self.bn_t(f_t)

        c_st = torch.einsum('bx,bx->x', f_s_norm, f_t_norm) / n
        c_diff = c_st - torch.ones_like(c_st)

        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(4.0)
        eps = 1e-5
        loss = torch.log(c_diff.sum() + eps)
        return loss


class Projector_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Projector_1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projector = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, fm):
        modified_fm = self.projector(fm)
        return modified_fm


class VID(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, init_var=5.0, eps=1e-6):
        super(VID, self).__init__()
        self.eps = eps
        self.regressor = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        ])

        self.alpha = nn.Parameter(
            np.log(np.exp(init_var-eps)-1.0) * torch.ones(out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
                # 	nn.init.constant_(m.weight, 1)
                # 	nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        pred_mean = self.regressor(fm_s)
        pred_var = torch.log(1.0 + torch.exp(self.alpha)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (torch.log(pred_var) + (pred_mean-fm_t)**2 / pred_var)
        loss = torch.mean(neg_log_prob)

        return loss


class CosineSimilarity(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarity, self).__init__()
        self.reduction = reduction

    def forward(self, student_afa_feature, teacher_feature):
        teacher_magnitude = torch.abs(torch.fft.fft2(teacher_feature))
        student_magnitude = torch.abs(torch.fft.fft2(student_afa_feature))

        student_magnitude = student_magnitude.view(student_magnitude.shape[0], student_magnitude.shape[1], -1)
        teacher_magnitude = teacher_magnitude.view(teacher_magnitude.shape[0], teacher_magnitude.shape[1], -1)

        cos_sim = F.cosine_similarity(student_magnitude, teacher_magnitude, dim=-1)  # (B, C)

        loss = 1 - cos_sim  # Cosine이 1이면 Loss=0

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction mode. Choose between 'mean' and 'sum'.")


class AngularDistance(nn.Module):
    def __init__(self, reduction='mean'):
        super(AngularDistance, self).__init__()
        self.reduction = reduction

    def forward(self, student_afa_feature, teacher_feature):
        teacher_magnitude = torch.abs(torch.fft.fft2(teacher_feature))
        student_magnitude = torch.abs(torch.fft.fft2(student_afa_feature))

        student_magnitude = student_magnitude.view(student_magnitude.shape[0], student_magnitude.shape[1], -1)
        teacher_magnitude = teacher_magnitude.view(teacher_magnitude.shape[0], teacher_magnitude.shape[1], -1)

        cos_sim = F.cosine_similarity(student_magnitude, teacher_magnitude, dim=-1)  # (B, C)
        angular_distance = torch.acos(torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)) / torch.pi  # 정규화 (0~1)

        if self.reduction == 'mean':
            return angular_distance.mean()
        elif self.reduction == 'sum':
            return angular_distance.sum()
        else:
            raise ValueError("Invalid reduction mode. Choose between 'mean' and 'sum'.")


class WKD(nn.Module):
    def __init__(self, wkd_level=3, wkd_basis='haar'):
        super(WKD, self).__init__()
        self.xfm = DWTForward(J=wkd_level, wave=wkd_basis, mode='zero')

    def forward(self, student, teacher):
        student_l, student_h = self.xfm(student)
        teacher_l, teacher_h = self.xfm(teacher)
        loss = 0.0
        for index in range(len(student_h)):
            loss += torch.nn.functional.l1_loss(teacher_h[index], student_h[index])
        return loss


class MLP_Module(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(MLP_Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        )

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()  # No pooling when pooling=False

    def forward(self, x):
        output = self.conv(x)
        output = self.pool(output)  # Apply pooling only if pooling=True

        return output


class Sobel_loss(nn.Module):
    def __init__(self):
        super(Sobel_loss, self).__init__()
        # Define Sobel kernels
        self.sobel_x = nn.Parameter(torch.tensor([[-1, 0, 1],
                                                  [-2, 0, 2],
                                                  [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                    requires_grad=False)  # (1, 1, 3, 3)
        self.sobel_y = nn.Parameter(torch.tensor([[-1, -2, -1],
                                                  [0, 0, 0],
                                                  [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                    requires_grad=False)  # (1, 1, 3, 3)

    def sobel_apply(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        # Compute gradient magnitude

        epsilon = 1e-8
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon)

        return grad_magnitude

    def forward(self, pred, gt):
        if pred.shape[1] == 3:
            pred_gray = TF.rgb_to_grayscale(pred, num_output_channels=1)  # (N, 1, H, W)
        else:
            pred_gray = pred

        if gt.shape[1] == 3:
            gt_gray = TF.rgb_to_grayscale(gt, num_output_channels=1)  # (N, 1, H, W)
        else:
            gt_gray = gt

        sobel_pred = self.sobel_apply(pred_gray)
        sobel_gt = self.sobel_apply(gt_gray)

        return F.l1_loss(sobel_pred, sobel_gt)

# class Phase_AFA_Module(nn.Module):
#     def __init__(self, in_channels, out_channels, shapes):
#         super(Phase_AFA_Module, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.shapes = shapes
#
#         # self.low_attention_weight = torch.nn.Parameter(torch.randn(self.out_channels, self.out_channels, 1, 1))
#         # self.high_attention_weight = torch.nn.Parameter(torch.randn(self.out_channels, self.out_channels, 1, 1))
#
#         self.low_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, groups=self.out_channels)
#         self.high_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, groups=self.out_channels)
#
#         self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1)
#
#     def forward(self, student_feature, teacher_phase):
#         cuton = 0.1
#
#         student_feature_1 = self.conv1(student_feature)
#         student_feature_relu = self.relu(student_feature_1)
#         student_feature_out = self.conv2(student_feature_relu)
#
#         x_ft = torch.fft.fft2(student_feature_out, norm="ortho")  # B, C, H, W
#         x_shift = torch.fft.fftshift(x_ft)  # B, C, H, W
#
#         magnitude = torch.abs(x_shift)  # 크기 값 계산
#         # phase = torch.angle(x_shift)  # 위상 값 계산
#
#         h, w = x_shift.shape[2:4]
#         cy, cx = int(h / 2), int(w / 2)
#         rh, rw = int(cuton * cy), int(cuton * cx)
#
#         # 전체를 먼저 0으로 초기화
#         low_pass = torch.zeros_like(magnitude)  # B, C, H, W
#         low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]
#
#         high_pass = magnitude - low_pass
#
#         # low_attn_map = F.conv2d(low_pass, self.low_attention_weight, padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
#         # high_attn_map = F.conv2d(high_pass, self.high_attention_weight, padding=0)
#
#         low_attn_map = self.low_conv(low_pass)  # channel 별 독립적인 conv
#         high_attn_map = self.high_conv(high_pass)
#
#         low_attn_map = torch.sigmoid(low_attn_map) + 0.5
#         high_attn_map = torch.sigmoid(high_attn_map) + 0.5
#
#         low_pass_att = low_attn_map * low_pass
#         high_pass_att = high_attn_map * high_pass
#
#         mag_out = low_pass_att + high_pass_att  # B, C, H, W
#
#         real = mag_out * torch.cos(teacher_phase)  # 실수부
#         imag = mag_out * torch.sin(teacher_phase)  # 허수부
#
#         fre_out = torch.complex(real, imag)
#
#         x_fft = torch.fft.ifftshift(fre_out)
#
#         out = torch.fft.ifft2(x_fft, s=(student_feature_out.size(-2), student_feature_out.size(-1)), norm="ortho").real
#
#         return out

# class AFA_Module(nn.Module):
#     def __init__(self, in_channels, out_channels, shapes):
#         super(AFA_Module, self).__init__()
#
#         """
#         feat_s_shape, feat_t_shape
#         2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
#         """
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.shapes = shapes
#
#         # self.rate1 = torch.nn.Parameter(torch.ones(1))
#         # self.rate2 = torch.nn.Parameter(torch.Tensor(1))
#
#         self.scale = (1 / (self.in_channels * self.out_channels))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes, dtype=torch.cfloat))
#
#         self.low_param = torch.nn.Parameter(torch.zeros(1))
#
#         # self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1)
#
#         self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
#         self.spatial_sigmoid = nn.Sigmoid()
#
#         # init_rate_half(self.rate1)
#         # init_rate_half(self.rate2)
#         # init_rate_half(self.low_param)
#
#         # 추가된 BatchNorm과 ReLU
#         # self.bn = nn.BatchNorm2d(self.out_channels)
#         # self.relu = nn.ReLU()
#
#     def compl_mul2d(self, input, weights):
#         return torch.einsum("bixy,ioxy->boxy", input, weights)
#
#     def spatial_attention(self, x):
#         # Compute mean and max along the channel dimension
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # 반환 값 values, indices
#         attention = torch.cat([avg_out, max_out], dim=1)
#         attention = self.spatial_conv(attention)
#         return self.spatial_sigmoid(attention)
#
#     def forward(self, x):
#         if isinstance(x, tuple):
#             x, cuton = x
#         else:
#             cuton = 0.1
#
#         batchsize = x.shape[0]
#         x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
#
#         # out_ft = self.compl_mul2d(x_ft, self.weights1)  # 복소수의 곱, channel 수 변환
#         # batch_fftshift = torch.fft.fftshift(out_ft)  # B, C, H, W
#
#         batch_fftshift = torch.fft.fftshift(x_ft)
#
#         # Magnitude와 Phase 분리
#         # mag_out = torch.abs(batch_fftshift)  # 크기 값 계산
#         magnitude = torch.abs(batch_fftshift)  # 크기 값 계산
#         phase = torch.angle(batch_fftshift)  # 위상 값 계산
#
#         # do the filter in here
#         h, w = batch_fftshift.shape[2:4]  # height and width
#         cy, cx = int(h / 2), int(w / 2)  # centerness
#         rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
#         # the value of center pixel is zero.
#
#         # 전체를 먼저 0으로 초기화
#         low_pass = torch.zeros_like(magnitude)  # B, C, H, W
#
#         # 저주파수 영역만 복사
#         low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]
#
#         # 고주파수 영역만 복사
#         high_pass = magnitude - low_pass
#
#         # 가중치를 제한하고 정규화된 가중합 적용
#         self.low_param.data.clamp_(min=-2.0, max=2.0)  # Sigmoid에 적합한 입력 범위로 제한
#
#         low_pass_weight = torch.sigmoid(self.low_param)
#         high_pass_weight = 1.0 - low_pass_weight  # 0 ~ 1 사이의 값
#
#         low_pass = low_pass * low_pass_weight
#         high_pass = high_pass * high_pass_weight
#
#         mag_out = low_pass + high_pass  # B, C, H, W
#
#         spatial_attention_map = self.spatial_attention(phase)
#         phase_out = phase * spatial_attention_map
#
#         # Magnitude와 Phase 결합 (복소수 생성)
#         real = mag_out * torch.cos(phase_out)  # 실수부
#         imag = mag_out * torch.sin(phase_out)  # 허수부
#
#         # 복소수 형태로 결합
#         fre_out = torch.complex(real, imag)
#
#         # batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0
#
#         # ifftshift 적용 (주파수 성분 복구)
#         x_fft = torch.fft.ifftshift(fre_out)
#
#         # Return to physical space
#         out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real
#
#         # self.rate1.data.clamp_(min=0.1, max=2.0)  # 최소 0.1, 최대 2.0으로 제한
#
#         # out2 = self.w0(x)
#
#         # epsilon = 1e-5
#         # out = torch.sigmoid(out) * self.rate1  # rate1이 0.1로 수렴
#         # out = torch.sigmoid(out) * x
#
#         # out2 = self.w0(x)
#         # out2 = self.rate2 * out2
#
#         # 최종 출력 계산 후 BatchNorm과 ReLU 적용
#         # output = x + out * self.rate1
#         output = x + torch.sigmoid(out) * x
#
#         # output = torch.cat([out, out2], dim=1)
#         # output = self.bn(output)
#         # output = self.relu(output)
#
#         return output
#
# def init_rate_half(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0.5)
#
# def init_rate_0(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0.)
