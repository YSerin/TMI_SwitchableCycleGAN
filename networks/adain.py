import torch
import torch.nn as nn


class half_PolyPhase_resUnet_Adain(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(half_PolyPhase_resUnet_Adain,self).__init__()
        self.inconv = inconv(in_ch = in_ch, out_ch = 64)
        self.down1 = down_half_polyphase(in_ch = 64*4, out_ch=128)
        self.down2 = down_half_polyphase(in_ch = 128*4, out_ch=256)
        self.down3 = down_half_polyphase(in_ch = 256*4, out_ch=512)
        self.down4 = down_half_polyphase_end(in_ch = 512*4, out_ch=1024)
        self.up4 = up_half_polyphase(in_ch= 1024 , out_ch = 512)
        self.up3 = up_half_polyphase(in_ch= 512 , out_ch = 256)
        self.up2 = up_half_polyphase(in_ch= 256 , out_ch = 128)
        self.up1 = up_half_polyphase(in_ch= 128 , out_ch = 64)
        self.outconv = outconv(in_ch = 64, out_ch=out_ch)

        self.adain_shared = adain_code_generator_shared()

    def forward(self,input, alpha):
        batch_size = input.shape[0]
        shared_code = self.adain_shared(batch_size)
        x0 = self.inconv(input, shared_code,  alpha)
        x1 = self.down1(x0, shared_code, alpha)
        x2 = self.down2(x1, shared_code, alpha)
        x3 = self.down3(x2, shared_code, alpha)
        x4 = self.down4(x3, shared_code, alpha)
        x = self.up4(x4, x3, shared_code, alpha)
        x = self.up3(x, x2, shared_code, alpha)
        x = self.up2(x, x1, shared_code, alpha)
        x = self.up1(x, x0, shared_code, alpha)
        x = self.outconv(x)
        output = input + x
        return output

class inconv(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv1 = one_conv(in_ch, out_ch)
        self.conv2 = one_conv_adain(out_ch, out_ch)

    def forward(self,x, shared_code, alpha):
        x = self.conv1(x)
        x = self.conv2(x, shared_code, alpha)
        return x

class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(one_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class one_conv_adain(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(one_conv_adain, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.instanceNorm = nn.InstanceNorm2d(out_ch)
        self.adain = adain_code_generator_seperate(out_ch)
        self.Leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self,x_in, shared_code, alpha):
        x_in = self.conv(x_in)
        x_in = self.instanceNorm(x_in)

        N, C, h, w = x_in.size()
        mean_y, sigma_y = self.adain(x_in, shared_code)
        x_out = sigma_y * (x_in) + mean_y

        x_out = x_out * (alpha) + x_in * (1 - alpha)
        x_out = self.Leakyrelu(x_out)
        return x_out


class down_half_polyphase(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(down_half_polyphase, self).__init__()
        self.conv1 = one_conv(in_ch, int(out_ch/2))
        self.conv2 = one_conv_adain(int(out_ch/2), out_ch)

    def forward(self,x, shared_code, alpha):
        x = subpixel_pooling(x)
        x = self.conv1(x)
        x = self.conv2(x, shared_code, alpha)
        return x

class down_half_polyphase_end(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(down_half_polyphase_end, self).__init__()
        self.conv1 = one_conv_adain(in_ch, int(out_ch/2))
        self.conv2 = one_conv_adain(int(out_ch/2), out_ch)

    def forward(self,x, shared_code, alpha):
        x = subpixel_pooling(x)
        x = self.conv1(x, shared_code, alpha)
        x = self.conv2(x, shared_code, alpha)
        return x

class up_half_polyphase(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_half_polyphase,self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=2, stride=2)
        self.conv1 = one_conv(in_ch, out_ch)
        self.conv2 = one_conv_adain(out_ch, out_ch)

    def forward(self, x1, x2, shared_code, alpha):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x, shared_code, alpha)
        return x


class outconv(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,1)

    def forward(self,x):
        x = self.conv(x)
        return x


def subpixel_pooling(x):
    x1 = x[:, :, ::2, ::2]
    x2 = x[:, :, ::2, 1::2]
    x3 = x[:, :, 1::2, ::2]
    x4 = x[:, :, 1::2, 1::2]

    out = torch.cat([x1,x2,x3,x4], dim=1)
    return out

class adain_code_generator_shared(nn.Module):
    def __init__(self):
        super(adain_code_generator_shared, self).__init__()
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,64)

    def forward(self, batch_size):
        self.ones_vec = torch.ones((batch_size, 128)).cuda(self.fc1.weight.device)

        x = self.fc1(self.ones_vec)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x       

class adain_code_generator_seperate(nn.Module):
    def __init__(self, ch):
        super(adain_code_generator_seperate, self).__init__()

        self.fc_mean = nn.Linear(64,ch)
        self.fc_var = nn.Linear(64,ch)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input, shared_code):
        N, C, h, w = input.size()

        fc_mean = self.fc_mean(shared_code)    
        fc_var = self.fc_var(shared_code)      
        fc_var = self.ReLU(fc_var)

        fc_mean_np = fc_mean.view(N,C,1,1).expand(N,C,h,w)     
        fc_var_np = fc_var.view(N,C,1,1).expand(N,C,h,w)      

        return fc_mean_np, fc_var_np




