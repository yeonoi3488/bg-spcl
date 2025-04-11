import torch
import torch.nn as nn
from torchsummary import summary
from models.layers import Conv2dWithConstraint
from einops import rearrange, reduce, repeat
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(self,
                temporal_size: int,
                num_channels: int,
                sampling_rate: int,
                is_gap: bool=False,
                dropout_rate: float=0.5):
        super(ShallowConvNet, self).__init__()
        
        kernel_size = int(sampling_rate * 0.16)
        pooling_kernel_size = int(sampling_rate * 0.3)
        pooling_stride_size = int(sampling_rate * 0.06)
        
        self.temporal_conv = Conv2dWithConstraint(1, 40, kernel_size=[1, kernel_size], padding='same', max_norm=2.)
        self.spatial_conv = Conv2dWithConstraint(40, 40, kernel_size=[num_channels, 1], padding='valid', max_norm=2.)
        self.bn = nn.BatchNorm2d(40)
        self.avg_pool = nn.AvgPool2d(kernel_size=[1, pooling_kernel_size], stride=[1, pooling_stride_size])
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        
    
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        
        x = self.flatten(x)
        x = self.dropout(x)
        return x       
    
    
if __name__ == '__main__':
    import argparse
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, nargs='+', default=[1, 64, 750])
    args = parser.parse_args()
    
    input_size = tuple(args.input_size)
    model = ShallowConvNet(input_size[-1], input_size[1])
    model.to(device)
    summary(model, input_size)