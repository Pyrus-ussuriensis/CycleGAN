import torch
import torch.nn as nn

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super(Resblock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=0, bias=True)
        self.innorm = nn.InstanceNorm2d(num_features=in_channels)
    def forward(self, x):
        y = self.padding1(x)
        y = self.conv1(y)
        y = self.innorm(y)
        return y

class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.resblock1 = Resblock(in_channels=in_channels)
        self.resblock2 = Resblock(in_channels=in_channels)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        y = self.resblock1(x)
        y = self.relu(y)
        #y = self.dropout(y)
        y = self.resblock2(y)
        return x + y

class INReLU(nn.Module):
    def __init__(self, in_channel):
        super(INReLU, self).__init__()
        self.innorm = nn.InstanceNorm2d(num_features=in_channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.innorm(x)
        y = self.relu(y)
        return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.padding1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.resnetblocks = nn.Sequential(*(ResNetBlock(in_channels=256) for _ in range(9)))
        self.convtrans1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.padding2 = nn.ReflectionPad2d(3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0)
        self.tanh = nn.Tanh()
        self.inrelu1 = INReLU(64)
        self.inrelu2 = INReLU(128)
        self.inrelu3 = INReLU(256)
        self.inrelu4 = INReLU(128)
        self.inrelu5 = INReLU(64)

    def forward(self, x):
        y = self.padding1(x)
        y = self.conv1(y); y = self.inrelu1(y)
        y = self.conv2(y); y = self.inrelu2(y)
        y = self.conv3(y); y = self.inrelu3(y)
        y = self.resnetblocks(y)
        y = self.convtrans1(y); y = self.inrelu4(y)
        y = self.convtrans2(y); y = self.inrelu5(y)
        y = self.padding2(y)
        y = self.conv4(y)
        y = self.tanh(y)
        return y




if __name__ == "__main__":
    x = torch.ones(size=(1, 3,256,256))
    generator = Generator()
    y = generator(x)
    print(y.size())
