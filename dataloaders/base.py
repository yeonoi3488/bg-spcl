from dataloaders.moabb import BNCI2014004


def get_dataset(args, is_test):
    
    args['is_test'] = True if is_test else False

    args.paradigm = 'MI'
    args.num_subjects = 9
    args.num_classes = 2
    args.sampling_rate = 250
    args.num_channels = 3
    args.trial_len = 4 
    args.temporal_size = 1126
    args.feature_deep_dim = 560

    dataloader = BNCI2014004(args)


    return dataloader