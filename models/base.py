import torch
import torch.nn as nn
from models.init import model_list
from models.layers import LazyLinearWithConstraint


class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()
        
        self.flatten = nn.Flatten()
        self.dense = LazyLinearWithConstraint(num_classes, max_norm=0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x


class Net(nn.Module):   
    def __init__(self,
                model_name: str,
                num_classes: int,
                num_channels: int,
                temporal_size: int,
                **kwargs):
        super(Net, self).__init__()

        self.backbone = model_list[model_name](temporal_size,
                                                num_channels,
                                                is_gap=False,
                                                **kwargs)

        self.classifier = classifier(num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def get_model(args):
    model_name = args.model
    if model_name == 'EGNet':
        model = Net(model_name=args.model, 
                    num_classes=args.num_classes,
                    num_channels=args.num_channels,
                    temporal_size=args.temporal_size,
                    n_fft=args.n_fft,
                    AE_mode=False,
                    use_EEGGram=True,
                    mask_ratio=0.3)
    elif model_name in ['ShallowConvNet', 'DeepConvNet', 'EEGNet',
                        'EEGTransformer']:
        model = Net(model_name=args.model,
                    num_classes=args.num_classes,
                    num_channels=args.num_channels,
                    temporal_size=args.temporal_size,
                    sampling_rate=args.sampling_rate)

    elif model_name in ['EEGResNet', 'EEGITNet']:
        model = Net(model_name=args.model, 
                    num_classes=args.num_classes,
                    num_channels=args.num_channels,
                    temporal_size=args.temporal_size)
        
    else:
        raise Exception('get_model function Wrong input!!!')

    return model
