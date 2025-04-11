import os, yaml
from easydict import EasyDict
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataloaders.base import get_dataset
from utils.setup_utils import get_device
from bg_spcl.trainers.offline import pretraining, eval
from bg_spcl.trainers.online import online_learning


'''Argparse'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='bnci2014004_config')
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--online_update', type=bool, default=False)
args = parser.parse_args()


# Config setting
with open(f'configs/{args.config_name}.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    cfg = EasyDict(config)

cfg['is_test'] = args.is_test
cfg['online_update'] = args.online_update

# Set device
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
cfg['device'] = get_device(args.gpu_num)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True


if __name__ == '__main__':

    trainset, testset = get_dataset(cfg, is_test=False)

    # Offline
    if not cfg.online_update:

        if not args.is_test:
            pretraining(cfg, trainset)

        else:
            test_acc_dict = eval(cfg, testset)
            print(test_acc_dict)
            print(f'Average test accuracy: {np.mean(list(test_acc_dict.values()))}')
            print(f'Std test accuracy: {np.std(list(test_acc_dict.values()))}')
    # Online
    else:
        online_learning(cfg, trainset, testset)