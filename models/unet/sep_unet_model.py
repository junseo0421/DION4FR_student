""" Full assembly of the parts to form the complete network """

from models.unet.sep_unet_parts import *


class Sep_UNet(nn.Module):  # m = 4
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Sep_UNet, self).__init__()
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
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.down6 = torch.utils.checkpoint(self.down6)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.up5 = torch.utils.checkpoint(self.up5)
        self.up6 = torch.utils.checkpoint(self.up6)
        self.outc = torch.utils.checkpoint(self.outc)

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
        return logits  # (2, 3, 192, 192)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class D_Thin_Sep_UNet_4(nn.Module):  # m = 4, feature 1/4
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
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class Q_Thin_Sep_UNet_4(nn.Module):  # m = 4, feature 1/8
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
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class Sep_UNet_6(nn.Module):  # m = 6, feature 1/2
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Sep_UNet_6, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = (DoubleConv(n_channels, 32))  # first two separable conv -> regular conv
        self.down1 = (SepDown(32, 64))
        self.down2 = (SepDown(64, 128))
        self.down3 = (SepDown(128, 256))
        self.down4 = (SepDown(256, 512))
        self.down5 = (SepDown(512, 1024))
        factor = 2 if bilinear else 1
        self.down6 = (SepDown(1024, 2048 // factor))  # bottleneck

        # decoder
        self.up1 = (SepUp(2048, 1024 // factor, bilinear))
        self.up2 = (SepUp(1024, 512 // factor, bilinear))
        self.up3 = (SepUp(512, 256 // factor, bilinear))
        self.up4 = (SepUp(256, 128 // factor, bilinear))
        self.up5 = (SepUp(128, 64 // factor, bilinear))
        self.up6 = (SepUp(64, 32, bilinear))

        # Output layer
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        # decoder
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)

        #output
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.down6 = torch.utils.checkpoint(self.down6)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.up5 = torch.utils.checkpoint(self.up5)
        self.up6 = torch.utils.checkpoint(self.up6)
        self.outc = torch.utils.checkpoint(self.outc)
