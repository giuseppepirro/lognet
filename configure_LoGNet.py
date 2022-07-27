"""Best hyperparameters found."""
import torch

LogNetConf = {
    'num_ew_channels': 8,
    'num_epochs': 2000,
    'in_feats': 1,
    'hidden_feats': 10,
    'out_feats': 112,      
    'n_layers': 3,
    'lr': 2e-2,
    'weight_decay': 0,
    'patience': 1000,
    'dropout': 0.2,           
    'aggr_mode': 'sum',     ## 'sum' or 'concat' for the aggregation across channels
    'ewnorm': 'both'
    }


def get_exp_configure(args):
    if (args['model'] == 'LogNetConf'):
        return LogNetConf
