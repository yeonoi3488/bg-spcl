import torch
import torch.nn as nn
from torchsummary import summary
from models.layers import Conv2dWithConstraint


class DeepConvNet(nn.Module):
    def __init__(self,
                temporal_size: int,
                num_channels: int,
                is_gap: bool=False,
                sampling_rate: int=250,
                dropout_rate: float=0.5):
        super(DeepConvNet, self).__init__()
        '''
        input size (1, 64, 375)일 때 오류 발생
        '''
        
        self.is_gap = is_gap
        
        kernel_size = int(sampling_rate * 0.02)
        self.temporal = Conv2dWithConstraint(1, 25, kernel_size=[1, kernel_size], padding='valid', max_norm=2.)
        self.spatial = Conv2dWithConstraint(25, 25, kernel_size=[num_channels, 1], padding='valid', max_norm=2.)
        
        self.block_1 = nn.Sequential(
            self.temporal,
            self.spatial,
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2]),
            nn.Dropout2d(dropout_rate)
        )
        
        self.block_2 = nn.Sequential(
            Conv2dWithConstraint(25, 50, kernel_size=[1, kernel_size], padding='valid', max_norm=2.),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2]),
            nn.Dropout2d(dropout_rate)
        )
        
        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(50, 100, kernel_size=[1, kernel_size], padding='valid', max_norm=2.),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2]),
            nn.Dropout2d(dropout_rate)
        )
        
        self.block_4 = nn.Sequential(
            Conv2dWithConstraint(100, 200, kernel_size=[1, kernel_size], padding='valid', max_norm=2.),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=[1, 3], stride=[1, 3]),
            nn.Dropout2d(dropout_rate)
        )
        
        if self.is_gap:
            self.gap = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        
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
    model = DeepConvNet(input_size[-1], input_size[1])
    model.to(device)
    summary(model, input_size)