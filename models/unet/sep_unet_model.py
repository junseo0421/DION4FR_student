""" Full assembly of the parts to form the complete network """

from models.unet.sep_unet_parts import *

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


# class DQ_Thin_Sep_UNet_4_Freq(nn.Module):  # m = 4, feature 1/8
#     def __init__(self, n_channels, n_classes, device, bilinear=False):
#         super(DQ_Thin_Sep_UNet_4_Freq, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         # Encoder
#         self.inc = (DoubleConv(n_channels, 4))  # first two separable conv -> regular conv
#         self.down1 = (SepDown(4, 8))
#         self.down2 = (SepDown(8, 16))
#         self.down3 = (SepDown(16, 32))
#         factor = 2 if bilinear else 1
#         self.down4 = (SepDown(32, 64 // factor))  # bottleneck

#         # decoder
#         self.up1 = (SepUp(64, 32 // factor, bilinear))
#         self.up2 = (SepUp(32, 16 // factor, bilinear))
#         self.up3 = (SepUp(16, 8 // factor, bilinear))
#         self.up4 = (SepUp(8, 4, bilinear))

#         # Output layer
#         self.outc = (OutConv(4, n_classes))

#         self.highpass1 = self.creathighpass(4, 4, device)
#         self.highpass2 = self.creathighpass(8, 8, device)
#         self.highpass3 = self.creathighpass(16, 16, device)
#         self.highpass4 = self.creathighpass(32, 32, device)

#         self.attention_weights1 = torch.nn.Parameter(torch.randn(4, 4, 1, 1))
#         self.attention_weights2 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))
#         self.attention_weights3 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))
#         self.attention_weights4 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))

#     def creathighpass(self, nchannel, outchannel, device):
#         high = torch.tensor([[0, -0.25, 0], [-0.25, 0, 0.25], [0, 0.25, 0]], dtype=torch.float32) 
#         high = high.unsqueeze(0).repeat(outchannel, 1, 1)
#         high = high.unsqueeze(0).repeat(nchannel, 1, 1, 1)
#         return high.to(device)

#     def forward(self, x):
#         relu = torch.nn.ReLU()
#         # sigma = nn.Sigmoid()

#         # Encoder
#         x1 = self.inc(x)
#         f_high1 = F.conv2d(x1, self.highpass1, padding=self.highpass1.size(-1) // 2)

#         x2 = self.down1(x1)
#         f_high2 = F.conv2d(x2, self.highpass2, padding=self.highpass2.size(-1) // 2)

#         x3 = self.down2(x2)
#         f_high3 = F.conv2d(x3, self.highpass3, padding=self.highpass3.size(-1) // 2)

#         x4 = self.down3(x3)
#         f_high4 = F.conv2d(x4, self.highpass4, padding=self.highpass4.size(-1) // 2)

#         x5 = self.down4(x4)

#         f_high4 = relu(f_high4)
#         attn_map4 = torch.nn.functional.conv2d(f_high4, self.attention_weights4, padding=0)
#         attn_map4 = torch.nn.functional.softmax(attn_map4, dim=1)
#         x4h = attn_map4 * x4

#         f_high3 = relu(f_high3)
#         attn_map3 = torch.nn.functional.conv2d(f_high3, self.attention_weights3, padding=0)
#         attn_map3 = torch.nn.functional.softmax(attn_map3, dim=1)
#         x3h = attn_map3 * x3

#         f_high2 = relu(f_high2)
#         attn_map2 = torch.nn.functional.conv2d(f_high2, self.attention_weights2, padding=0)
#         attn_map2 = torch.nn.functional.softmax(attn_map2, dim=1)
#         x2h = attn_map2 * x2

#         f_high1 = relu(f_high1)
#         attn_map1 = torch.nn.functional.conv2d(f_high1, self.attention_weights1, padding=0)
#         attn_map1 = torch.nn.functional.softmax(attn_map1, dim=1)
#         x1h = attn_map1 * x1

#         # decoder
#         x = self.up1(x5, x4 + x4h)
#         x = self.up2(x, x3 + x3h)
#         x = self.up3(x, x2 + x2h)
#         x = self.up4(x, x1 + x1h)

#         # # decoder
#         # x = self.up1(x5, x4h)
#         # x = self.up2(x, x3h)
#         # x = self.up3(x, x2h)
#         # x = self.up4(x, x1h)

#         #output
#         logits = self.outc(x)
#         return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


 class Sep_UNet_4_AFA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Sep_UNet_4_AFA, self).__init__()
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

        self.low_attention_weights1 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.randn(128, 128, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.randn(128, 128, 1, 1))

        self.low_attention_weights4 = torch.nn.Parameter(torch.randn(256, 256, 1, 1))
        self.high_attention_weights4 = torch.nn.Parameter(torch.randn(256, 256, 1, 1))

        self.concat_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft)  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight, padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out)

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
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class D_Sep_UNet_4_AFA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(D_Sep_UNet_4_AFA, self).__init__()
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

        self.low_attention_weights1 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))
        self.high_attention_weights1 = torch.nn.Parameter(torch.randn(16, 16, 1, 1))

        self.low_attention_weights2 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))
        self.high_attention_weights2 = torch.nn.Parameter(torch.randn(32, 32, 1, 1))

        self.low_attention_weights3 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))
        self.high_attention_weights3 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))

        self.low_attention_weights4 = torch.nn.Parameter(torch.randn(128, 128, 1, 1))
        self.high_attention_weights4 = torch.nn.Parameter(torch.randn(128, 128, 1, 1))

        self.concat_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.concat_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.concat_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.concat_conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft)  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight, padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out)

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
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}


class DQ_Thin_Sep_UNet_4_AFA(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DQ_Thin_Sep_UNet_4_AFA, self).__init__()
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

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)

    def afa_module(self, x, low_attention_weight, high_attention_weight):  # x는 0 ~ 1 값
        cuton = 0.1
        x_ft = torch.fft.fft2(x, norm="ortho")  # B, C, H, W
        x_shift = torch.fft.fftshift(x_ft)  # B, C, H, W

        magnitude = torch.abs(x_shift)  # 크기 값 계산
        phase = torch.angle(x_shift)  # 위상 값 계산

        h, w = x_shift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size

        low_pass = torch.zeros_like(magnitude)  # B, C, H, W
        low_pass[:, :, cy - rh:cy + rh, cx - rw:cx + rw] = magnitude[:, :, cy - rh:cy + rh, cx - rw:cx + rw]

        high_pass = magnitude - low_pass

        low_attn_map = torch.nn.functional.conv2d(low_pass, low_attention_weight, padding=0)  # (B, C, H, W)에 (C, C, 1, 1)의 weight 로 conv 적용 (1x1 conv)
        high_attn_map = torch.nn.functional.conv2d(high_pass, high_attention_weight, padding=0)

        low_attn_map = torch.nn.functional.softmax(low_attn_map, dim=1)
        high_attn_map = torch.nn.functional.softmax(high_attn_map, dim=1)

        low_pass_att = low_attn_map * low_pass
        high_pass_att = high_attn_map * high_pass

        mag_out = low_pass_att + high_pass_att

        real = mag_out * torch.cos(phase)  # 실수부
        imag = mag_out * torch.sin(phase)  # 허수부

        fre_out = torch.complex(real, imag)

        x_fft = torch.fft.ifftshift(fre_out)

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
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}

