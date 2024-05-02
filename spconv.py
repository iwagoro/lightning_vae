import torch.nn as nn
class sub_pixel_conv(nn.Module):
    def __init__(self,in_channels,out_channels,r=1):
        super(sub_pixel_conv,self).__init__()
        self.conv =  nn.Conv2d(in_channels,out_channels*r*r ,kernel_size=1)
        self.shuffle = nn.PixelShuffle(r)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x
    