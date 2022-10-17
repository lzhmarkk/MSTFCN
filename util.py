import os
import sys
import math
import json
import h5py
import types
import torch
import numpy as np
import pandas as pd
from model.esg import ESG
from baselines import *
from data.dataloader import StandardScaler


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def get_config():
    with open(f"./config.json", 'r') as f:
        run_cfg = json.load(f)
    model, dataset = run_cfg['model_name'], run_cfg['data']
    with open(f"./data/config/{dataset}.json", 'r') as f:
        data_cfg = json.load(f)

    if model == 'ESG':
        from model.model_config import config as model_cfg  # load model config
    elif model == 'DCRNN':
        with open(f"./baselines/DCRNN/config.json", 'r') as f:
            model_cfg = json.load(f)
    elif model == 'GMAN':
        with open(f"./baselines/GMAN/config.json", 'r') as f:
            model_cfg = json.load(f)
    elif model == 'GWNet':
        with open(f"./baselines/GWNet/config.json", 'r') as f:
            model_cfg = json.load(f)
    else:
        raise ValueError()

    cfg = types.SimpleNamespace()
    for c in [run_cfg, model_cfg, data_cfg]:
        for k, v in c.items():
            setattr(cfg, k, v)
    return cfg


def get_auxiliary(args, dataloader):
    ret = {}
    if args.model_name == 'ESG':
        node_fea = get_node_fea(args.data, 0.7)
        node_fea = torch.tensor(node_fea).type(torch.FloatTensor).to(args.device)
        ret['node_fea'] = node_fea
        ret['fc_dim'] = (dataloader['train_loader'].size - 8) * 16 * 2
    elif args.model_name == 'DCRNN':
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mx'] = np.array(df['adjacency_matrix'])
        ret['num_batches'] = math.ceil(len(df['raw_data']) / args.batch_size)
    elif args.model_name == 'GMAN':
        from baselines.GMAN.SE import load_se_file
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mx'] = np.array(df['adjacency_matrix'])
        se = load_se_file(adj_mx=ret['adj_mx'],
                          adj_file=os.path.join('./baselines/GMAN', args.data + '_adj.edgelist'),
                          se_file=os.path.join('./baselines/GMAN', args.data + '_se'))
        ret['se'] = np.repeat(se, args.input_dim, axis=0)
        # ret['se'] = se
    elif args.model_name == 'GWNet':
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mx'] = np.array(df['adjacency_matrix'])
    return ret


def get_model(args):
    if args.model_name == 'ESG':
        model = ESG(args.dy_embedding_dim, args.dy_interval, args.num_nodes, args.window, args.horizon, args.input_dim,
                    args.output_dim, 1, args.layers, conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels=args.end_channels, kernel_set=args.kernel_set,
                    dilation_exp=args.dilation_exponential, gcn_depth=args.gcn_depth,
                    device=args.device, fc_dim=args.fc_dim, st_embedding_dim=args.st_embedding_dim,
                    dropout=args.dropout, propalpha=args.propalpha, layer_norm_affline=False,
                    static_feat=args.node_fea)
    elif args.model_name == 'DCRNN':
        model = DCRNN(adj_mx=args.adj_mx, num_batches=args.num_batches, device=args.device, max_diffusion_step=args.max_diffusion_step,
                      cl_decay_steps=args.step_size, filter_type=args.filter_type, num_nodes=args.num_nodes,
                      num_rnn_layers=args.num_rnn_layers, rnn_units=args.rnn_units, use_curriculum_learning=args.cl,
                      input_dim=args.input_dim, window=args.window, output_dim=args.output_dim, horizon=args.horizon)
    elif args.model_name == 'GMAN':
        model = GMAN(SE=args.se, device=args.device, L=args.L, K=args.K, d=args.d, bn_decay=args.bn_decay,
                     window=args.window, input_dim=args.input_dim, output_dim=args.output_dim)
    elif args.model_name == 'GWNet':
        model = GWNet(adj_mx=args.adj_mx, device=args.device, adjtype=args.adjtype, randomadj=args.randomadj, aptonly=args.aptonly,
                      nhid=args.nhid, input_dim=args.input_dim, output_dim=args.output_dim, num_nodes=args.num_nodes,
                      kernel_size=args.kernel_size, horizon=args.horizon, window=args.window, dropout=args.dropout,
                      blocks=args.blocks, layers=args.layers, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj)
    else:
        raise ValueError()
    return model


def get_node_fea(data_set, train_num=0.6):
    if data_set == 'solar-energy':
        path = 'data/h5data/solar-energy.h5'
    elif data_set == 'electricity':
        path = 'data/h5data/electricity.h5'
    elif data_set == 'exchange-rate':
        path = 'data/h5data/exchange-rate.h5'
    elif data_set == 'wind':
        path = 'data/h5data/wind.h5'
    elif data_set == 'nyc-bike':
        path = 'data/h5data/nyc-bike.h5'
    elif data_set == 'nyc-taxi':
        path = 'data/h5data/nyc-taxi.h5'
    else:
        raise ('No such dataset........................................')

    if data_set == 'nyc-bike' or data_set== 'nyc-taxi':
        x = h5py.File(path, 'r')
        data = np.array(x['raw_data'])
        data = data.transpose([0, 2, 1])  # (T, C, N)
        df = data[:int(len(data) * train_num)]
        scaler = StandardScaler(df.mean(),df.std())
        train_feas = scaler.transform(df).reshape([-1,df.shape[2]])
    else:
        x = pd.read_hdf(path)
        data = x.values
        print(x.shape)
        num_samples = data.shape[0]
        num_train = round(num_samples * train_num)
        df = data[:num_train]
        print(df.shape)
        scaler = StandardScaler(df.mean(),df.std())
        train_feas = scaler.transform(df)
    return train_feas


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
