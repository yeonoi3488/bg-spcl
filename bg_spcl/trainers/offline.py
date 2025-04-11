import glob
import numpy as np
import pandas as pd
from easydict import EasyDict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import get_model
from bg_spcl.spcl import main_spcl


def pretraining(args, dataset):

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset,
                                batch_size=len(dataset),
                                pin_memory=False,
                                sampler=train_subsampler)
        val_loader = DataLoader(dataset,
                                batch_size=args.train_batch,
                                pin_memory=False,
                                sampler=val_subsampler)
        
        model = get_model(args).to(args.device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_loss = 1000

        for epoch in range(args.EPOCHS):

            for i, (x, y, c) in enumerate(train_loader):
                main_spcl(x, y, c, model, criterion, optimizer, args)
            
            model.eval()
            y_true, y_pred = [], []
            
            for x, y, c in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss = loss.item()
                
                _, predict = torch.max(outputs, 1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(predict.cpu().tolist())

            acc = accuracy_score(y_true, y_pred)
                
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict()
        
        print(f'Fold: {fold}, Loss: {val_loss}, Acc: {acc}')            
        fname = args.LOG_NAME + f'P{args.target_subject}_fold_{fold}_best_model.pth'
        torch.save(best_model, fname)


def eval(args, dataset):

    test_acc_dict = {}
    for fold in range(args.k_folds):
        fname = args.LOG_NAME + f'P{args.target_subject}_fold_{fold}_best_model.pth'
        model_path = glob.glob(fname)[0]

        model = get_model(args)
        model.load_state_dict(torch.load(model_path))
        model = model.to(args.device)
        model.eval()

        test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        y_true, y_pred = [], []
        for x, y, c in test_loader:
            x, y = x.to(args.device), y.to(args.device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            y_true.append(y.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

        acc = accuracy_score(y_true, y_pred)
        test_acc_dict[fold] = acc

    return test_acc_dict






