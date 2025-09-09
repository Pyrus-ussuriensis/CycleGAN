import torch
import torch.nn as nn

class ConvNormLReLU(nn.Module):
    def __init__(self, in_channel, stride):
        super(ConvNormLReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2, kernel_size=4, stride=stride, padding=1, bias=True)
        self.norm = nn.InstanceNorm2d(num_features=in_channel*2)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.LReLU(y)
        return y

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.cnlr1 = ConvNormLReLU(64, stride=2)
        self.cnlr2 = ConvNormLReLU(128, stride=2)
        self.cnlr3 = ConvNormLReLU(256, stride=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
    def forward(self, x):
        y = self.conv1(x)
        y = self.lrelu(y)
        y = self.cnlr1(y)
        y = self.cnlr2(y)
        y = self.cnlr3(y)
        y = self.conv2(y)
        return y

if __name__ == "__main__":
    x = torch.ones(size=(1,3,256,256))
    discriminator = Discriminator()
    y = discriminator(x)
    print(y.size())

