import torch
import torch.nn as nn
from torchsummary import summary
from models.layers import Conv2dWithConstraint


class EEGNet(nn.Module):
    def __init__(self,
                temporal_size: int,
                num_channels: int,
                is_gap: bool=False,
                dropout_rate: float=0.5,
                F1: int=8,
                F2: int=16,
                D: int=2,
                sampling_rate: int=250,
                ):
        super(EEGNet, self).__init__()
        
        self.is_gap = is_gap
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=[1, sampling_rate // 2], padding='same', bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, D * F1, kernel_size=[num_channels, 1], groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=[1, 4], stride=[1, 4]),
            nn.Dropout2d(dropout_rate)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(D * F1, F2, kernel_size=[1, sampling_rate // 8], padding='same', groups=D * F1, bias=False), 
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=[1, 8], stride=[1, 8]),
            nn.Dropout2d(dropout_rate)
        )
        
        if self.is_gap:
            self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        
        if self.is_gap:
            x = self.gap(x)
        
        return x
    
    
if __name__ == '__main__':
    import argparse
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, nargs='+', default=[1, 64, 750])
    args = parser.parse_args()
    
    input_size = tuple(args.input_size)
    model = EEGNet(input_size[-1], input_size[1])
    model.to(device)
    summary(model, input_size)