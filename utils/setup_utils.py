import torch


def get_device(GPU_NUM: str) -> torch.device:
    if torch.cuda.device_count() == 1:
        output = torch.device('cuda')
    elif torch.cuda.device_count() > 1:
        output = torch.device(f'cuda:{GPU_NUM}')
    else:
        output = torch.device('cpu')

    return output

def get_log_name(args):

    log_list = {
        'subject': args.target_subject,
        'model': args.model,
        'epoch': args.EPOCHS,
        'batch_size': args.train_batch,
        'lr': args.lr,
        'spcl': args.spcl,
    }

    output = log_list['time']
    for key in log_list.keys():
        if key != 'time':
            output += f'_{key}_{log_list[key]}'

    print(f'Log name: \n\t{output}')

    return output