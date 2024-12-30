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


class DQ_Thin_Sep_UNet_4_Freq(nn.Module):  # m = 4, feature 1/8
    def __init__(self, n_channels, n_classes, device, bilinear=False):
        super(DQ_Thin_Sep_UNet_4_Freq, self).__init__()
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

        self.highpass1 = self.creathighpass(4, 4, device)
        self.highpass2 = self.creathighpass(8, 8, device)

        self.attention_weights1 = torch.nn.Parameter(torch.randn(4, 4, 1, 1))
        self.attention_weights2 = torch.nn.Parameter(torch.randn(8, 8, 1, 1))

    def creathighpass(self, nchannel, outchannel, device):
        high = torch.tensor([[0, -0.25, 0], [-0.25, 0, -0.25], [0, -0.25, 0]], dtype=torch.float32)  ### make it 1
        high = high.unsqueeze(0).repeat(outchannel, 1, 1)
        high = high.unsqueeze(0).repeat(nchannel, 1, 1, 1)
        return high.to(device)

    def forward(self, x):
        relu = torch.nn.ReLU()
        # sigma = nn.Sigmoid()

        # Encoder
        x1 = self.inc(x)
        f_high1 = F.conv2d(x1, self.highpass1, padding=self.highpass1.size(-1) // 2)

        x2 = self.down1(x1)
        f_high2 = F.conv2d(x2, self.highpass2, padding=self.highpass1.size(-1) // 2)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        f_high2 = relu(f_high2)
        attn_map2 = torch.nn.functional.conv2d(f_high2, self.attention_weights2, padding=0)
        attn_map2 = torch.nn.functional.softmax(attn_map2, dim=1)
        x2h = attn_map2 * x2

        f_high1 = relu(f_high1)
        attn_map1 = torch.nn.functional.conv2d(f_high1, self.attention_weights1, padding=0)
        attn_map1 = torch.nn.functional.softmax(attn_map1, dim=1)
        x1h = attn_map1 * x1

        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2 + x2h)
        x = self.up4(x, x1 + x1h)

        #output
        logits = self.outc(x)
        return logits, {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}
    