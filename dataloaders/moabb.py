import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch 
from torch.utils.data import TensorDataset

from bg_spcl.bg import compute_bg_scores
from preprocessing import bandpass_filter, standardize_data


def BNCI2014004(args):

    X = np.load(f'data/{args.data}/X.npy')
    y = np.load(f'data/{args.data}/labels.npy')
    meta = pd.read_csv(f'data/{args.data}/meta.csv')

    train_indices = [
        i for i, (session, subject) in enumerate(zip(meta['session'], meta['subject'])) 
        if session in ['0train', '1train', '2train'] and subject == args.target_subject + 1
    ]
    test_indices = [
        i for i, (session, subject) in enumerate(zip(meta['session'], meta['subject'])) 
        if session in ['3test', '4test'] and subject == args.target_subject + 1
    ]


    X_train= X[train_indices]
    y_train= y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]


    # Apply bandpass filtering (except for specified models)
    if args.model not in ['EEGNet', 'EEGITNet', 'EISATC_Fusion']:
        lowcut, highcut, fs = 4, 38, 250  # Define filter parameters
        X_train = np.array([bandpass_filter(sample, lowcut, highcut, fs) for sample in X_train])
        X_test = np.array([bandpass_filter(sample, lowcut, highcut, fs) for sample in X_test])

    # Z-score Normalization (Channel-wise standardization)
    X_train, X_test = standardize_data(X_train, X_test)

    # Expand dimensions for CNN models
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Label encoding
    le = preprocessing.LabelEncoder()
    y_train = torch.tensor(le.fit_transform(y_train), dtype=torch.long)
    y_test = torch.tensor(le.fit_transform(y_test), dtype=torch.long)

    # Compute brain-guided (BG) scores
    c_train = torch.tensor(compute_bg_scores(args, X_train), dtype=torch.float32)
    c_test = torch.tensor(compute_bg_scores(args, X_test), dtype=torch.float32)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Adjust shape based on model type
    if args.model == 'EEGConformer':
        X_train, X_test = X_train.squeeze(-1), X_test.squeeze(-1)
    elif args.model != 'EEGITNet':  # Permute for CNN models
        X_train, X_test = X_train.permute(0, 3, 1, 2), X_test.permute(0, 3, 1, 2)

    # Move tensors to the specified device
    device = args.device
    X_train, y_train, c_train = X_train.to(device), y_train.to(device), c_train.to(device)
    X_test, y_test, c_test = X_test.to(device), y_test.to(device), c_test.to(device)

    # Print data shapes for verification
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, c_train shape: {c_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, c_test shape: {c_test.shape}")

    # Create PyTorch datasets
    train_data = TensorDataset(X_train, y_train, c_train)
    test_data = TensorDataset(X_test, y_test, c_test)

    return train_data, test_data

