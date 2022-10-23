import os
import sys
import math
import json
import h5py
import types
import torch
import random
import numpy as np
from baselines import *


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

    with open(f"./baselines/{model}/config.json", 'r') as f:
        model_cfg = json.load(f)

    cfg = types.SimpleNamespace()
    for c in [run_cfg, model_cfg, data_cfg]:
        for k, v in c.items():
            setattr(cfg, k, v)
    return cfg


def get_auxiliary(args, dataloader):
    ret = {}
    if args.model_name == 'ESG':
        from baselines.ESG.esg_utils import get_node_fea, get_fc_dim
        node_fea = get_node_fea(args.data, 0.7)
        ret['node_fea'] = torch.tensor(node_fea).type(torch.FloatTensor).to(args.device)
        ret['fc_dim'] = get_fc_dim(dataloader['train_loader'].size, args.residual_channels)
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
    elif args.model_name == "MTGNN":
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mx'] = np.array(df['adjacency_matrix'])
    elif args.model_name == "STGCN":
        from baselines.STGCN.norm_adj_mx import get_normalized_adj
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        adj_mx = np.array(df['adjacency_matrix'])
        ret['adj_mx'] = get_normalized_adj(adj_mx)
    elif args.model_name == "STID":
        pass
    else:
        raise ValueError(f"Auxiliary data for model {args.model_name} is not found")
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
    elif args.model_name == 'MTGNN':
        model = MTGNN(device=args.device, adj_mx=args.adj_mx, gcn_true=args.gcn_true, buildA_true=args.buildA_true,
                      num_nodes=args.num_nodes, gcn_depth=args.gcn_depth, dropout=args.dropout,
                      input_dim=args.input_dim, output_dim=args.output_dim, window=args.window, horizon=args.horizon,
                      subgraph_size=args.subgraph_size, node_dim=args.node_dim, tanhalpha=args.tanhalpha, propalpha=args.propalpha,
                      dilation_exponential=args.dilation_exponential, layers=args.layers, residual_channels=args.residual_channels,
                      conv_channels=args.conv_channels, skip_channels=args.skip_channels, end_channels=args.end_channels)
    elif args.model_name == "STID":
        model = STID(device=args.device, num_nodes=args.num_nodes, node_dim=args.node_dim, window=args.window, horizon=args.horizon,
                     input_dim=args.input_dim, output_dim=args.output_dim, embed_dim=args.embed_dim,
                     num_mlp_layers=args.num_mlp_layers, temp_dim_tid=args.temp_dim_tid, temp_dim_diw=args.temp_dim_diw)
    elif args.model_name == 'STGCN':
        model = STGCN(adj_mx=args.adj_mx, device=args.device, num_nodes=args.num_nodes, input_dim=args.input_dim,
                      output_dim=args.output_dim, window=args.window, horizon=args.horizon)
    else:
        raise ValueError(f"Model {args.model_name} is not found")
    return model


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
