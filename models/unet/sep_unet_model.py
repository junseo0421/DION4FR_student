""" Full assembly of the parts to form the complete network """
import torch.nn

from models.unet.sep_unet_parts import *


class Sep_UNet_Feature(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Sep_UNet_Feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 64))  # first two separable conv -> regular conv
        self.down1 = (SepDown(64, 128))
        self.down2 = (SepDown(128, 256))
        self.down3 = (SepDown(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(512, 1024 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(1024, 512 // factor, bilinear))
        self.up2 = (SepUp(512, 256 // factor, bilinear))
        self.up3 = (SepUp(256, 128 // factor, bilinear))
        self.up4 = (SepUp(128, 64, bilinear))

        # Output layer
        self.outc = (OutConv(64, n_classes))

        self.low_attention_weights1 = nn.Parameter(torch.empty(64, 64, 1, 1))
        self.high_attention_weights1 = nn.Parameter(torch.empty(64, 64, 1, 1))

        self.low_attention_weights2 = nn.Parameter(torch.empty(128, 128, 1, 1))
        self.high_attention_weights2 = nn.Parameter(torch.empty(128, 128, 1, 1))

        self.low_attention_weights3 = nn.Parameter(torch.empty(256, 256, 1, 1))
        self.high_attention_weights3 = nn.Parameter(torch.empty(256, 256, 1, 1))

        self.low_attention_weights4 = nn.Parameter(torch.empty(512, 512, 1, 1))
        self.high_attention_weights4 = nn.Parameter(torch.empty(512, 512, 1, 1))

        # kaiming uniform 초기화
        nn.init.kaiming_uniform_(self.low_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights4, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights4, a=0, mode='fan_in', nonlinearity='relu')

        self.concat_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-4)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-4)
        self.bn4 = nn.BatchNorm2d(512, eps=1e-4)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        magnitude = magnitude.clamp(min=1e-6, max=1e6)

        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight, padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
        high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att
        mag_out = mag_out.clamp(min=1e-6, max=1e6)

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real  # phase는 그대로 사용

        return out

    def forward(self, x):
        # Encoder
        # 각각 relu 취해져서 나옴
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_out = self.afa_module(x4, self.low_attention_weights4, self.high_attention_weights4)
        x4_concat = torch.cat((x4, x4_out), dim=1)  # Concatenate along the channel dimension
        x4_output = self.concat_conv4(x4_concat)  # Reduce back to original channels
        x4_output = self.bn4(x4_output)  # BatchNormalization
        x4_output = self.relu(x4_output)  # ReLU

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)  # Concatenate along the channel dimension
        x3_output = self.concat_conv3(x3_concat)  # Reduce back to original channels
        x3_output = self.bn3(x3_output)  # BatchNormalization
        x3_output = self.relu(x3_output)  # ReLU

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)  # Concatenate along the channel dimension
        x2_output = self.concat_conv2(x2_concat)  # Reduce back to original channels
        x2_output = self.bn2(x2_output)  # BatchNormalization
        x2_output = self.relu(x2_output)  # ReLU

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)  # Concatenate along the channel dimension
        x1_output = self.concat_conv1(x1_concat)  # Reduce back to original channels
        x1_output = self.bn1(x1_output)  # BatchNormalization
        x1_output = self.relu(x1_output)  # ReLU

        # decoder
        x = self.up1(x5, x4_output)
        x = self.up2(x, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4, "x4_out": x4_output, "x5": x5}


class Thin_Sep_UNet_4_Feature(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Thin_Sep_UNet_4_Feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 32))  # first two separable conv -> regular conv
        self.down1 = (SepDown(32, 64))
        self.down2 = (SepDown(64, 128))
        self.down3 = (SepDown(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(256, 512 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(512, 256 // factor, bilinear))
        self.up2 = (SepUp(256, 128 // factor, bilinear))
        self.up3 = (SepUp(128, 64 // factor, bilinear))
        self.up4 = (SepUp(64, 32, bilinear))

        # Output layer
        self.outc = (OutConv(32, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.empty(32, 32, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.empty(32, 32, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.empty(64, 64, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.empty(64, 64, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.empty(128, 128, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.empty(128, 128, 1, 1))

        self.low_attention_weights4 = torch.nn.Parameter(torch.empty(256, 256, 1, 1))
        self.high_attention_weights4 = torch.nn.Parameter(torch.empty(256, 256, 1, 1))

        nn.init.kaiming_uniform_(self.low_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights4, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights4, a=0, mode='fan_in', nonlinearity='relu')

        self.concat_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-4)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-4)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-4)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        magnitude = magnitude.clamp(min=1e-6, max=1e6)

        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight,
                                                  padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
        high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att
        mag_out = mag_out.clamp(min=1e-6, max=1e6)

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real  # phase는 그대로 사용

        return out

    def forward(self, x):
        # Encoder
        # 각각 relu 취해져서 나옴
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_out = self.afa_module(x4, self.low_attention_weights4, self.high_attention_weights4)
        x4_concat = torch.cat((x4, x4_out), dim=1)  # Concatenate along the channel dimension
        x4_output = self.concat_conv4(x4_concat)  # Reduce back to original channels
        x4_output = self.bn4(x4_output)  # BatchNormalization
        x4_output = self.relu(x4_output)  # ReLU

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)  # Concatenate along the channel dimension
        x3_output = self.concat_conv3(x3_concat)  # Reduce back to original channels
        x3_output = self.bn3(x3_output)  # BatchNormalization
        x3_output = self.relu(x3_output)  # ReLU

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)  # Concatenate along the channel dimension
        x2_output = self.concat_conv2(x2_concat)  # Reduce back to original channels
        x2_output = self.bn2(x2_output)  # BatchNormalization
        x2_output = self.relu(x2_output)  # ReLU

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)  # Concatenate along the channel dimension
        x1_output = self.concat_conv1(x1_concat)  # Reduce back to original channels
        x1_output = self.bn1(x1_output)  # BatchNormalization
        x1_output = self.relu(x1_output)  # ReLU

        # decoder
        x = self.up1(x5, x4_output)
        x = self.up2(x, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4, "x4_out": x4_output, "x5": x5}


class D_Thin_Sep_UNet_4_Feature(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(D_Thin_Sep_UNet_4_Feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 16))  # first two separable conv -> regular conv
        self.down1 = (SepDown(16, 32))
        self.down2 = (SepDown(32, 64))
        self.down3 = (SepDown(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(128, 256 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(256, 128 // factor, bilinear))
        self.up2 = (SepUp(128, 64 // factor, bilinear))
        self.up3 = (SepUp(64, 32 // factor, bilinear))
        self.up4 = (SepUp(32, 16, bilinear))

        # Output layer
        self.outc = (OutConv(16, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.empty(16, 16, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.empty(16, 16, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.empty(32, 32, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.empty(32, 32, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.empty(64, 64, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.empty(64, 64, 1, 1))

        self.low_attention_weights4 = torch.nn.Parameter(torch.empty(128, 128, 1, 1))
        self.high_attention_weights4 = torch.nn.Parameter(torch.empty(128, 128, 1, 1))

        # kaiming uniform 초기화
        nn.init.kaiming_uniform_(self.low_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights4, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights4, a=0, mode='fan_in', nonlinearity='relu')

        self.concat_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(16, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-4)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-4)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-4)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        magnitude = magnitude.clamp(min=1e-6, max=1e6)

        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight,
                                                  padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
        high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att
        mag_out = mag_out.clamp(min=1e-6, max=1e6)

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real  # phase는 그대로 사용

        return out

    def forward(self, x):
        # Encoder
        # 각각 relu 취해져서 나옴
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_out = self.afa_module(x4, self.low_attention_weights4, self.high_attention_weights4)
        x4_concat = torch.cat((x4, x4_out), dim=1)  # Concatenate along the channel dimension
        x4_output = self.concat_conv4(x4_concat)  # Reduce back to original channels
        x4_output = self.bn4(x4_output)  # BatchNormalization
        x4_output = self.relu(x4_output)  # ReLU

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)  # Concatenate along the channel dimension
        x3_output = self.concat_conv3(x3_concat)  # Reduce back to original channels
        x3_output = self.bn3(x3_output)  # BatchNormalization
        x3_output = self.relu(x3_output)  # ReLU

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)  # Concatenate along the channel dimension
        x2_output = self.concat_conv2(x2_concat)  # Reduce back to original channels
        x2_output = self.bn2(x2_output)  # BatchNormalization
        x2_output = self.relu(x2_output)  # ReLU

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)  # Concatenate along the channel dimension
        x1_output = self.concat_conv1(x1_concat)  # Reduce back to original channels
        x1_output = self.bn1(x1_output)  # BatchNormalization
        x1_output = self.relu(x1_output)  # ReLU

        # decoder
        x = self.up1(x5, x4_output)
        x = self.up2(x, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4, "x4_out": x4_output, "x5": x5}


class Q_Thin_Sep_UNet_4_Feature(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Q_Thin_Sep_UNet_4_Feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 8))  # first two separable conv -> regular conv
        self.down1 = (SepDown(8, 16))
        self.down2 = (SepDown(16, 32))
        self.down3 = (SepDown(32, 64))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(64, 128 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(128, 64 // factor, bilinear))
        self.up2 = (SepUp(64, 32 // factor, bilinear))
        self.up3 = (SepUp(32, 16 // factor, bilinear))
        self.up4 = (SepUp(16, 8, bilinear))

        # Output layer
        self.outc = (OutConv(8, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))

        self.low_attention_weights4 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))
        self.high_attention_weights4 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))

        self.concat_conv1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(8, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-4)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-4)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-4)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        magnitude = magnitude.clamp(min=1e-6, max=1e6)

        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight,
                                                  padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
        high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att
        mag_out = mag_out.clamp(min=1e-6, max=1e6)

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real  # phase는 그대로 사용

        return out

    def forward(self, x):
        # Encoder
        # 각각 relu 취해져서 나옴
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_out = self.afa_module(x4, self.low_attention_weights4, self.high_attention_weights4)
        x4_concat = torch.cat((x4, x4_out), dim=1)  # Concatenate along the channel dimension
        x4_output = self.concat_conv4(x4_concat)  # Reduce back to original channels
        x4_output = self.bn4(x4_output)  # BatchNormalization
        x4_output = self.relu(x4_output)  # ReLU

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)  # Concatenate along the channel dimension
        x3_output = self.concat_conv3(x3_concat)  # Reduce back to original channels
        x3_output = self.bn3(x3_output)  # BatchNormalization
        x3_output = self.relu(x3_output)  # ReLU

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)  # Concatenate along the channel dimension
        x2_output = self.concat_conv2(x2_concat)  # Reduce back to original channels
        x2_output = self.bn2(x2_output)  # BatchNormalization
        x2_output = self.relu(x2_output)  # ReLU

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)  # Concatenate along the channel dimension
        x1_output = self.concat_conv1(x1_concat)  # Reduce back to original channels
        x1_output = self.bn1(x1_output)  # BatchNormalization
        x1_output = self.relu(x1_output)  # ReLU

        # decoder
        x = self.up1(x5, x4_output)
        x = self.up2(x, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4, "x4_out": x4_output, "x5": x5}


class DQ_Thin_Sep_UNet_4_Feature(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DQ_Thin_Sep_UNet_4_Feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 4))  # first two separable conv -> regular conv
        self.down1 = (SepDown(4, 8))
        self.down2 = (SepDown(8, 16))
        self.down3 = (SepDown(16, 32))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(32, 64 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(64, 32 // factor, bilinear))
        self.up2 = (SepUp(32, 16 // factor, bilinear))
        self.up3 = (SepUp(16, 8 // factor, bilinear))
        self.up4 = (SepUp(8, 4, bilinear))

        # Output layer
        self.outc = (OutConv(4, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.randn(4, 4, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.randn(4, 4, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))

        self.low_attention_weights4 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))
        self.high_attention_weights4 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))

        self.concat_conv1 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(4, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-4)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-4)
        self.bn4 = nn.BatchNorm2d(32, eps=1e-4)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        magnitude = magnitude.clamp(min=1e-6, max=1e6)

        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight,
                                                  padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
        high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att
        mag_out = mag_out.clamp(min=1e-6, max=1e6)

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real  # phase는 그대로 사용

        return out

    def forward(self, x):
        # Encoder
        # 각각 relu 취해져서 나옴
        x1 = self.inc(x)  # (B, 4, 192, 192)
        x2 = self.down1(x1)   # (B, 8, 96, 96)
        x3 = self.down2(x2)  # (B, 16, 48, 48)
        x4 = self.down3(x3)  # (B, 32, 24, 24)
        x5 = self.down4(x4)   # (B, 64, 12, 12)

        x4_out = self.afa_module(x4, self.low_attention_weights4, self.high_attention_weights4)
        x4_concat = torch.cat((x4, x4_out), dim=1)  # Concatenate along the channel dimension
        x4_output = self.concat_conv4(x4_concat)  # Reduce back to original channels
        x4_output = self.bn4(x4_output)  # BatchNormalization
        x4_output = self.relu(x4_output)  # ReLU

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)  # Concatenate along the channel dimension
        x3_output = self.concat_conv3(x3_concat)  # Reduce back to original channels
        x3_output = self.bn3(x3_output)  # BatchNormalization
        x3_output = self.relu(x3_output)  # ReLU

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)  # Concatenate along the channel dimension
        x2_output = self.concat_conv2(x2_concat)  # Reduce back to original channels
        x2_output = self.bn2(x2_output)  # BatchNormalization
        x2_output = self.relu(x2_output)  # ReLU

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)  # Concatenate along the channel dimension
        x1_output = self.concat_conv1(x1_concat)  # Reduce back to original channels
        x1_output = self.bn1(x1_output)  # BatchNormalization
        x1_output = self.relu(x1_output)  # ReLU

        # decoder
        x = self.up1(x5, x4_output)
        x = self.up2(x, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4, "x4_out": x4_output, "x5": x5}


class V_Thin_Sep_UNet_4_Feature(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(V_Thin_Sep_UNet_4_Feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 4))  # first two separable conv -> regular conv
        self.down1 = (SepDown(4, 8))
        self.down2 = (SepDown(8, 16))
        self.down3 = (SepDown(16, 32))
        factor = 2 if bilinear else 1

        # decoder
        self.up2 = (SepUp(32, 16 // factor, bilinear))
        self.up3 = (SepUp(16, 8 // factor, bilinear))
        self.up4 = (SepUp(8, 4, bilinear))

        # Output layer
        self.outc = (OutConv(4, n_classes))

        self.low_attention_weights1 = torch.nn.Parameter(torch.empty(4, 4, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.empty(4, 4, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.empty(8, 8, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.empty(8, 8, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.empty(16, 16, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.empty(16, 16, 1, 1))

        nn.init.kaiming_uniform_(self.low_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights1, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights2, a=0, mode='fan_in', nonlinearity='relu')

        nn.init.kaiming_uniform_(self.low_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.high_attention_weights3, a=0, mode='fan_in', nonlinearity='relu')

        self.concat_conv1 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(4, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-4)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-4)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft, dim=(-2, -1))  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        magnitude = magnitude.clamp(min=1e-6, max=1e6)

        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight,
                                                  padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = low_attn_map - low_attn_map.amax(dim=1, keepdim=True)
        high_attn_map = high_attn_map - high_attn_map.amax(dim=1, keepdim=True)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att
        mag_out = mag_out.clamp(min=1e-6, max=1e6)

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out, dim=(-2, -1))

        out = torch.fft.ifft2(x_fft, s=(x.size(-2), x.size(-1)), norm="ortho").real  # phase는 그대로 사용

        return out

    def forward(self, x):
        # Encoder
        # 각각 relu 취해져서 나옴
        x1 = self.inc(x)  # (B, 4, 192, 192)
        x2 = self.down1(x1)   # (B, 8, 96, 96)
        x3 = self.down2(x2)  # (B, 16, 48, 48)
        x4 = self.down3(x3)  # (B, 32, 24, 24)

        x3_out = self.afa_module(x3, self.low_attention_weights3, self.high_attention_weights3)
        x3_concat = torch.cat((x3, x3_out), dim=1)  # Concatenate along the channel dimension
        x3_output = self.concat_conv3(x3_concat)  # Reduce back to original channels
        x3_output = self.bn3(x3_output)  # BatchNormalization
        x3_output = self.relu(x3_output)  # ReLU

        x2_out = self.afa_module(x2, self.low_attention_weights2, self.high_attention_weights2)
        x2_concat = torch.cat((x2, x2_out), dim=1)  # Concatenate along the channel dimension
        x2_output = self.concat_conv2(x2_concat)  # Reduce back to original channels
        x2_output = self.bn2(x2_output)  # BatchNormalization
        x2_output = self.relu(x2_output)  # ReLU

        x1_out = self.afa_module(x1, self.low_attention_weights1, self.high_attention_weights1)
        x1_concat = torch.cat((x1, x1_out), dim=1)  # Concatenate along the channel dimension
        x1_output = self.concat_conv1(x1_concat)  # Reduce back to original channels
        x1_output = self.bn1(x1_output)  # BatchNormalization
        x1_output = self.relu(x1_output)  # ReLU

        # decoder
        x = self.up2(x4, x3_output)
        x = self.up3(x, x2_output)
        x = self.up4(x, x1_output)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x1_out": x1_output, "x2": x2, "x2_out": x2_output, "x3": x3, "x3_out": x3_output, "x4": x4}


class Sep_UNet_4(nn.Module):  # m = 4, feature 1/2
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Sep_UNet_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 32))  # first two separable conv -> regular conv
        self.down1 = (SepDown(32, 64))
        self.down2 = (SepDown(64, 128))
        self.down3 = (SepDown(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(256, 512 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(512, 256 // factor, bilinear))
        self.up2 = (SepUp(256, 128 // factor, bilinear))
        self.up3 = (SepUp(128, 64 // factor, bilinear))
        self.up4 = (SepUp(64, 32, bilinear))

        # Output layer
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class D_Thin_Sep_UNet_4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(D_Thin_Sep_UNet_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 16))  # first two separable conv -> regular conv
        self.down1 = (SepDown(16, 32))
        self.down2 = (SepDown(32, 64))
        self.down3 = (SepDown(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(128, 256 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(256, 128 // factor, bilinear))
        self.up2 = (SepUp(128, 64 // factor, bilinear))
        self.up3 = (SepUp(64, 32 // factor, bilinear))
        self.up4 = (SepUp(32, 16, bilinear))

        # Output layer
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}



class Q_Thin_Sep_UNet_4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Q_Thin_Sep_UNet_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 8))  # first two separable conv -> regular conv
        self.down1 = (SepDown(8, 16))
        self.down2 = (SepDown(16, 32))
        self.down3 = (SepDown(32, 64))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(64, 128 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(128, 64 // factor, bilinear))
        self.up2 = (SepUp(64, 32 // factor, bilinear))
        self.up3 = (SepUp(32, 16 // factor, bilinear))
        self.up4 = (SepUp(16, 8, bilinear))

        # Output layer
        self.outc = (OutConv(8, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}



class DQ_Thin_Sep_UNet_4(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DQ_Thin_Sep_UNet_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 4))  # first two separable conv -> regular conv
        self.down1 = (SepDown(4, 8))
        self.down2 = (SepDown(8, 16))
        self.down3 = (SepDown(16, 32))
        factor = 2 if bilinear else 1
        self.down4 = (SepDown(32, 64 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(64, 32 // factor, bilinear))
        self.up2 = (SepUp(32, 16 // factor, bilinear))
        self.up3 = (SepUp(16, 8 // factor, bilinear))
        self.up4 = (SepUp(8, 4, bilinear))

        # Output layer
        self.outc = (OutConv(4, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class V_Thin_Sep_UNet_4(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(V_Thin_Sep_UNet_4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 4))  # first two separable conv -> regular conv
        self.down1 = (SepDown(4, 8))
        self.down2 = (SepDown(8, 16))
        self.down3 = (SepDown(16, 32))
        factor = 2 if bilinear else 1
        # self.down4 = (SepDown(32, 64 // factor))  # bottleneck

        # decoder
        # self.up1 = (SepUp(64, 32 // factor, bilinear))
        self.up2 = (SepUp(32, 16 // factor, bilinear))
        self.up3 = (SepUp(16, 8 // factor, bilinear))
        self.up4 = (SepUp(8, 4, bilinear))

        # Output layer
        self.outc = (OutConv(4, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # decoder
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        # x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4}

