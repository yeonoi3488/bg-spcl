import argparse
import numpy as np
import yaml, random
from easydict import EasyDict
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from models.base import get_model


def incre_update(args, test_time, data_cum, state_scores, model, cali_loader, optimizer, criterion):

    i = test_time
    X_buffer, y_buffer = [], [] # size is args.buffer_size !!!(nb)

    ''' Make pseudol abels '''
    batch_test = data_cum[i - args.buffer_size + 1:i + 1].numpy() 
    batch_test = batch_test.reshape(args.buffer_size, 1, batch_test.shape[2], batch_test.shape[3]) 
    batch_test = torch.from_numpy(batch_test).to(torch.float32).to(args.device)

    batch_scores = state_scores[i - args.buffer_size + 1:i + 1] 
    batch_scores = torch.tensor(batch_scores) 

    outputs = model(batch_test)
    softmax_out = nn.Softmax(dim=1)(outputs)
    best_confidence_values, _ = torch.max(softmax_out, dim=1)
    pseudo_labels = torch.argmax(softmax_out, dim=1)

    # sorting 50 % high confidence samples # Why needed ? in batch update ;
    _, indices = torch.sort(best_confidence_values, descending=True)
    indices = indices[:int(args.buffer_size - args.replay_ratio)]
    X_online = batch_test[indices] # EEG
    y_online = pseudo_labels[indices]

    X_offline, y_offline, c_offline = source_data_generator(cali_loader)
    
    X_buffer = torch.cat((X_offline, X_online), 0)
    y_buffer = torch.cat((y_offline, y_online), 0)
    
    # Update the model
    optimizer.zero_grad()
    outputs = model(X_buffer)
    loss = criterion(outputs, y_buffer)
    loss.backward()
    optimizer.step()
    model.eval()


def online_learning(args, cali_loader, test_loader): 

    # load the pre-trained params
    model = get_model(args).to(args.device)
    model.load_state_dict(torch.load(args.LOG_NAME)) 
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()   

    y_true, y_pred, s_scores = [], [], []
    for i, (x,y,c) in enumerate(test_loader):
        model.eval() 
        
        s_scores.append(c.item())
        x = x.reshape(1, 1, x.shape[-2], x.shape[-1]).cpu()

        if i == 0:
            data_cum = x.float().cpu()
        else:
            data_cum = torch.cat((data_cum, x.float().cpu()), 0) # concat EEG data

        sample_test = data_cum[i].numpy()
        sample_test = sample_test.reshape(1, 1, sample_test.shape[1], sample_test.shape[2])
        sample_test = torch.from_numpy(sample_test).to(torch.float32).to(args.device)

        outputs = model(sample_test)
        softmax_out = nn.Softmax(dim=1)(outputs)
        
        y_pred.append(softmax_out.detach().cpu().numpy())
        y_true.append(y.item()) 

        if args.online_update:
            model.train() # train mode
            
            if (i + 1) >= args.buffer_size*(1-args.replay_ratio) and (i + 1) % args.stride == 0: # buffer size is mini-batch size
                incre_update(args, i, data_cum, s_scores, model, cali_loader, optimizer, criterion)

    _, predict = torch.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.num_classes ), 1)
    pred = torch.squeeze(predict).float()
    acc = accuracy_score(y_true, pred)
    torch.save(model.state_dict(), f'{args.LOG_PATH}/online_update.pth')


def source_data_generator(offline_loader):
    data_list = list(offline_loader)    
    i = random.randint(0, len(data_list) - 1)
    x, y, c = data_list[i]
    
    print(f'generated source data : {i}, {len(x)}')
    return x, y, c