import torch.nn as nn


class Discriminator2(nn.Module):
    def __init__(self, in_ch=1, out_ch=64):
        super(Discriminator2, self).__init__()
        self.inconv = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2)
        self.LeakyRelu = nn.LeakyReLU(negative_slope=0.2)
        self.disc_CBR1 = disc_CBR(out_ch, out_ch*2, kernel_size=5, stride=2)
        self.disc_CBR2 = disc_CBR(out_ch*2, out_ch*4, kernel_size=5, stride=1)
        self.disc_CBR3 = disc_CBR(out_ch*4, out_ch*8, kernel_size=5, stride=1)
        self.outconv = nn.Conv2d(out_ch*8, 1, kernel_size=5, stride=1, padding=True)

    def forward(self, x):
       
        x = self.inconv(x)      
        x = self.LeakyRelu(x)   
        x = self.disc_CBR1(x)   
        x = self.disc_CBR2(x)   
        x = self.disc_CBR3(x)   
        x = self.outconv(x)     

        return x       

class disc_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=1):
        super(disc_CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1, stride=stride, bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self, x):
        x = self.block(x)
        return x




