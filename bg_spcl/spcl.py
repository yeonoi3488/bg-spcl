import torch
import torch.nn as nn
import math
import numpy as np
import wandb


def sample_align(softmax_output, labels, round, args):

    class_groups = {}
    tmplosses = []
    for i, data in enumerate(softmax_output):

        label = labels[i].item()

        pmax, predict = torch.max(data, 0)
        predict = predict.item()
        acc = 1 if label == predict else -1
        tmploss = 1 - pmax * acc
        tmplosses.append(tmploss.item())


        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append((tmploss, i))

    # easy sample indices
    sample_indices = {}
    for label, group in class_groups.items():
        group = sorted(group)
        spcl_lambda = args.init_lambda + args.spcl_lr * round

        count = int(len(group) * spcl_lambda)
        sample_indices[label] = [idx for _, idx in group[:count]]

    
    selected_indicies = []
    for indicies in sample_indices.values():
        selected_indicies.extend(indicies)

    return selected_indicies


def masking(args, k, c):
    Q1 = np.percentile(c, 25)
    Q3 = np.percentile(c, 75)
    IQR = Q3 - Q1

    outliers = c > (Q3 + k * IQR)

    mask = ~outliers
    args['outliers'] = outliers.sum()

    return mask


def main_spcl(x, y, c, model, criterion, optimizer, args): 

    c = c.float().cpu()
    mask = masking(args, args.k, c)
    x, y, c = x[mask], y[mask], c[mask]

    selected_indices = []
    for r in range(args.spcl_round):
        outputs = model(x)
        outputs = outputs.float().cpu()
        args.epsilon = 1e-5
        softmax_out = nn.Softmax(dim=1)(outputs)
        selected_indices = sample_align(softmax_out, y, r, args)    
        if len(selected_indices) != 0:
            outputs = model(x[selected_indices])
            loss = criterion(outputs, y[selected_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


