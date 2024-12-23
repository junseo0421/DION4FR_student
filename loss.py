# from pytorch_wavelets import DWTForward, DWTInverse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


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

# # 24.10.15 FSP Loss
# class FSP(nn.Module):
#     def __init__(self):
#         super(FSP, self).__init__()

#     def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
#         loss = F.mse_loss(self.fsp_matrix(fm_s1, fm_s2), self.fsp_matrix(fm_t1, fm_t2))

#         return loss

#     def fsp_matrix(self, fm1, fm2):
#         if fm1.size(2) > fm2.size(2):
#             fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

#         fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)  # B, C, H, W
#         fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

#         fsp = torch.bmm(fm1, fm2) / fm1.size(2)

#         return fsp
    
# class AFA_Module_cat(nn.Module):
#     def __init__(self, in_channels, out_channels, shapes):
#         super(AFA_Module_cat, self).__init__()

#         """
#         feat_s_shape, feat_t_shape
#         2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#         """
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.shapes = shapes
#         #  print(self.shapes)
#         self.rate1 = torch.nn.Parameter(torch.Tensor(1))
#         self.rate2 = torch.nn.Parameter(torch.Tensor(1))
#         # self.out_channels = feat_t_shape[1]
#         self.scale = (1 / (self.in_channels * self.out_channels))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(self.in_channels, self.out_channels // 2, self.shapes, self.shapes, dtype=torch.cfloat))

#         self.low_param = torch.nn.Parameter(torch.Tensor(1))

#         self.w0 = nn.Conv2d(self.in_channels, self.out_channels // 2, 1)

#         self.spatial_conv = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, bias=False)
#         self.spatial_sigmoid = nn.Sigmoid()

#         init_rate_half(self.rate1)
#         init_rate_half(self.rate2)
#         init_rate_half(self.low_param)

#         # 추가된 BatchNorm과 ReLU
#         self.bn = nn.BatchNorm2d(self.out_channels)
#         self.relu = nn.ReLU()

#     def compl_mul2d(self, input, weights):
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def spatial_attention(self, x):
#         # Compute mean and max along the channel dimension
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # 반환 값 values, indices
#         attention = torch.cat([avg_out, max_out], dim=1)
#         attention = self.spatial_conv(attention)
#         return self.spatial_sigmoid(attention)

#     def forward(self, x):
#         if isinstance(x, tuple):
#             x, cuton = x
#         else:
#             cuton = 0.1

#         batchsize = x.shape[0]
#         x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W

#         #  print(x_ft.shape)
#         out_ft = self.compl_mul2d(x_ft, self.weights1)  # 복소수의 곱, channel 수 변환
#         batch_fftshift = torch.fft.fftshift(out_ft)  # B, C, H, W

#         # Magnitude와 Phase 분리
#         # mag_out = torch.abs(batch_fftshift)  # 크기 값 계산
#         magnitude = torch.abs(batch_fftshift)  # 크기 값 계산
#         phase = torch.angle(batch_fftshift)  # 위상 값 계산

#         # do the filter in here
#         h, w = batch_fftshift.shape[2:4]  # height and width
#         cy, cx = int(h / 2), int(w / 2)  # centerness
#         rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
#         # the value of center pixel is zero.

#         # 전체를 먼저 0으로 초기화
#         low_pass = torch.zeros_like(magnitude)  # B, C, H, W

#         # 저주파수 영역만 복사
#         low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

#         # 고주파수 영역만 복사
#         high_pass = magnitude - low_pass

#         # 가중치를 제한하고 정규화된 가중합 적용
#         low_pass_weight = 0.5 + torch.sigmoid(self.low_param)
#         high_pass_weight = 2.0 - low_pass_weight  # 0.5 ~ 1.5 사이의 값

#         low_pass = low_pass * low_pass_weight
#         high_pass = high_pass * high_pass_weight

#         mag_out = low_pass + high_pass  # B, C, H, W

#         spatial_attention_map = self.spatial_attention(phase)
#         phase_out = phase * spatial_attention_map

#         # Magnitude와 Phase 결합 (복소수 생성)
#         real = mag_out * torch.cos(phase_out)  # 실수부
#         imag = mag_out * torch.sin(phase_out)  # 허수부

#         # 복소수 형태로 결합
#         fre_out = torch.complex(real, imag)

#         # batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0

#         # ifftshift 적용 (주파수 성분 복구)
#         x_fft = torch.fft.ifftshift(fre_out)

#         # Return to physical space
#         out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real

#         out2 = self.w0(x)

#         # 최종 출력 계산 후 BatchNorm과 ReLU 적용
#         # output = self.rate1 * out + self.rate2 * out2
#         output = torch.cat([out, out2], dim=1)
#         output = self.bn(output)
#         output = self.relu(output)

#         return output

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


class AFA_Module(nn.Module):
    def __init__(self, in_channels, out_channels, shapes):
        super(AFA_Module, self).__init__()

        """
        feat_s_shape, feat_t_shape
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes

        self.rate1 = torch.nn.Parameter(torch.ones(1))
        # self.rate2 = torch.nn.Parameter(torch.Tensor(1))

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes, dtype=torch.cfloat))

        self.low_param = torch.nn.Parameter(torch.zeros(1))

        # self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1)

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

        # init_rate_half(self.rate1)
        # init_rate_half(self.rate2)
        # init_rate_half(self.low_param)

        # 추가된 BatchNorm과 ReLU
        # self.bn = nn.BatchNorm2d(self.out_channels)
        # self.relu = nn.ReLU()

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def spatial_attention(self, x):
        # Compute mean and max along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 반환 값 values, indices
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.spatial_conv(attention)
        return self.spatial_sigmoid(attention)

    def forward(self, x):
        if isinstance(x, tuple):
            x, cuton = x
        else:
            cuton = 0.1

        batchsize = x.shape[0]
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W

        #  print(x_ft.shape)
        out_ft = self.compl_mul2d(x_ft, self.weights1)  # 복소수의 곱, channel 수 변환
        batch_fftshift = torch.fft.fftshift(out_ft)  # B, C, H, W

        # Magnitude와 Phase 분리
        # mag_out = torch.abs(batch_fftshift)  # 크기 값 계산
        magnitude = torch.abs(batch_fftshift)  # 크기 값 계산
        phase = torch.angle(batch_fftshift)  # 위상 값 계산

        # do the filter in here
        h, w = batch_fftshift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
        # the value of center pixel is zero.

        # 전체를 먼저 0으로 초기화
        low_pass = torch.zeros_like(magnitude)  # B, C, H, W

        # 저주파수 영역만 복사
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        # 고주파수 영역만 복사
        high_pass = magnitude - low_pass

        # 가중치를 제한하고 정규화된 가중합 적용
        self.low_param.data.clamp_(min=-2.0, max=2.0)  # Sigmoid에 적합한 입력 범위로 제한

        low_pass_weight = torch.sigmoid(self.low_param)
        high_pass_weight = 1.0 - low_pass_weight  # 0 ~ 1 사이의 값

        low_pass = low_pass * low_pass_weight
        high_pass = high_pass * high_pass_weight

        mag_out = low_pass + high_pass  # B, C, H, W

        spatial_attention_map = self.spatial_attention(phase)
        phase_out = phase * spatial_attention_map

        # Magnitude와 Phase 결합 (복소수 생성)
        real = mag_out * torch.cos(phase_out)  # 실수부
        imag = mag_out * torch.sin(phase_out)  # 허수부

        # 복소수 형태로 결합
        fre_out = torch.complex(real, imag)

        # batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0

        # ifftshift 적용 (주파수 성분 복구)
        x_fft = torch.fft.ifftshift(fre_out)

        # Return to physical space
        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real

        self.rate1.data.clamp_(min=0.1, max=2.0)  # 최소 0.1, 최대 2.0으로 제한

        # epsilon = 1e-5
        # out = torch.sigmoid(out) * self.rate1  # rate1이 0.1로 수렴
        out = torch.sigmoid(out) * x  

        # out2 = self.w0(x)
        # out2 = self.rate2 * out2

        # 최종 출력 계산 후 BatchNorm과 ReLU 적용
        output = x + out * self.rate1
        # output = torch.cat([out, out2], dim=1)
        # output = self.bn(output)
        # output = self.relu(output)

        return output


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)
