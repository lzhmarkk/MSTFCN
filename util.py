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
        self.log = open(fileN, "w")

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
    elif args.model_name == 'GWNetMix':
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mxs'] = [np.array(df['adjacency_matrix'])] * 2
    elif args.model_name == "MTGNN":
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mx'] = np.array(df['adjacency_matrix'])
    elif args.model_name == "STGCN":
        from baselines.STGCN.norm_adj_mx import get_normalized_adj
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        adj_mx = np.array(df['adjacency_matrix'])
        ret['adj_mx'] = get_normalized_adj(adj_mx)
    elif args.model_name == 'MOHER':
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mx'] = np.array(df['adjacency_matrix'])
        ret['node_fea'] = None
    elif args.model_name == 'STCGNN':
        from baselines.STCGNN.util import gen_Ac
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['As'] = np.array(df['adjacency_matrix'])
        ret['Ac'] = gen_Ac(np.array(df['raw_data']))
    elif args.model_name == 'STSHN':
        from baselines.STSHN.norm_adj import norm_adj
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj'] = norm_adj(np.array(df['adjacency_matrix']))
    elif args.model_name == 'CoGNN':
        df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
        ret['adj_mx'] = np.array(df['adjacency_matrix'])
    return ret


def get_model(args):
    if args.model_name == 'ESG':
        model = ESG(args.dy_embedding_dim, args.dy_interval, args.num_nodes, args.window, args.horizon, args.input_dim,
                    args.output_dim, 1, args.layers, conv_channels=args.conv_channels,
                    residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels=args.end_channels, kernel_set=args.kernel_set,
                    dilation_exp=args.dilation_exponential, gcn_depth=args.gcn_depth,
                    device=args.device, fc_dim=args.fc_dim, st_embedding_dim=args.st_embedding_dim,
                    dropout=args.dropout, propalpha=args.propalpha, layer_norm_affline=False,
                    static_feat=args.node_fea)
    elif args.model_name == 'DCRNN':
        model = DCRNN(adj_mx=args.adj_mx, device=args.device,
                      max_diffusion_step=args.max_diffusion_step,
                      num_nodes=args.num_nodes,
                      num_rnn_layers=args.num_rnn_layers, rnn_units=args.rnn_units,
                      input_dim=args.input_dim, window=args.window, output_dim=args.output_dim, horizon=args.horizon)
    elif args.model_name == 'GMAN':
        model = GMAN(SE=args.se, device=args.device, L=args.L, K=args.K, d=args.d, bn_decay=args.bn_decay,
                     window=args.window, input_dim=args.input_dim, output_dim=args.output_dim)
    elif args.model_name == 'GWNet':
        model = GWNet(adj_mx=args.adj_mx, device=args.device, adjtype=args.adjtype, randomadj=args.randomadj,
                      aptonly=args.aptonly,
                      nhid=args.nhid, input_dim=args.input_dim, output_dim=args.output_dim, num_nodes=args.num_nodes,
                      kernel_size=args.kernel_size, horizon=args.horizon, window=args.window, dropout=args.dropout,
                      blocks=args.blocks, layers=args.layers, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj)
    elif args.model_name == 'GWNetMix':
        model = GWNetMix(adj_mxs=args.adj_mxs, device=args.device, adjtype=args.adjtype, randomadj=args.randomadj,
                         aptonly=args.aptonly, nhid=args.nhid, input_dim=args.input_dim, output_dim=args.output_dim,
                         num_nodes=args.num_nodes, kernel_size=args.kernel_size, horizon=args.horizon,
                         window=args.window, dropout=args.dropout, blocks=args.blocks, layers=args.layers,
                         add_time=args.add_time, subgraph_size=args.subgraph_size, node_dim=args.node_dim,
                         tanhalpha=args.tanhalpha)
    elif args.model_name == 'MLPMixer':
        model = MLPMixer(device=args.device, input_dim=args.input_dim, output_dim=args.output_dim, window=args.window,
                         horizon=args.horizon, hidden_dim=args.hidden_dim, dropout=args.dropout,
                         num_nodes=args.num_nodes,
                         use_fft=args.use_fft, fft_dropout=args.fft_dropout)
    elif args.model_name == 'MTGNN':
        model = MTGNN(device=args.device, adj_mx=args.adj_mx, gcn_true=args.gcn_true, buildA_true=args.buildA_true,
                      num_nodes=args.num_nodes, gcn_depth=args.gcn_depth, dropout=args.dropout,
                      input_dim=args.input_dim, output_dim=args.output_dim, window=args.window, horizon=args.horizon,
                      subgraph_size=args.subgraph_size, node_dim=args.node_dim, tanhalpha=args.tanhalpha,
                      propalpha=args.propalpha,
                      dilation_exponential=args.dilation_exponential, layers=args.layers,
                      residual_channels=args.residual_channels,
                      conv_channels=args.conv_channels, skip_channels=args.skip_channels,
                      end_channels=args.end_channels,
                      add_time=args.add_time)
    elif args.model_name == 'MTGNNMix':
        model = MTGNNMix(device=args.device, num_nodes=args.num_nodes, gcn_depth=args.gcn_depth,
                         dropout=args.dropout, input_dim=args.input_dim, output_dim=args.output_dim,
                         window=args.window, horizon=args.horizon, subgraph_size=args.subgraph_size,
                         node_dim=args.node_dim, tanhalpha=args.tanhalpha, propalpha=args.propalpha,
                         dilation_exponential=args.dilation_exponential, layers=args.layers,
                         residual_channels=args.residual_channels,
                         conv_channels=args.conv_channels, skip_channels=args.skip_channels,
                         end_channels=args.end_channels,
                         add_time=args.add_time)
    elif args.model_name == 'FCGAGA':
        model = FCGAGA(device=args.device, n_stacks=args.n_stacks, n_blocks=args.n_blocks,
                       block_layers=args.block_layers, hidden_units=args.hidden_units,
                       node_id_dim=args.node_id_dim, input_dim=args.input_dim, output_dim=args.output_dim,
                       num_nodes=args.num_nodes, window=args.window, horizon=args.horizon, epsilon=args.epsilon)
    elif args.model_name == 'MSTFCN':
        model = MSTFCN(device=args.device, num_nodes=args.num_nodes, gcn_depth=args.gcn_depth, dropout=args.dropout,
                      input_dim=args.input_dim, output_dim=args.output_dim, window=args.window, horizon=args.horizon,
                      subgraph_size=args.subgraph_size, node_dim=args.node_dim, tanhalpha=args.tanhalpha,
                      propalpha=args.propalpha, add_time=args.add_time,
                      layers=args.layers, residual_channels=args.residual_channels,
                      conv_channels=args.conv_channels, skip_channels=args.skip_channels,
                      end_channels=args.end_channels, cross=args.cross, temporal_func=args.temporal_func)
    elif args.model_name == "STID":
        model = STID(device=args.device, num_nodes=args.num_nodes, node_dim=args.node_dim, window=args.window,
                     horizon=args.horizon,
                     input_dim=args.input_dim, output_dim=args.output_dim, embed_dim=args.embed_dim,
                     num_mlp_layers=args.num_mlp_layers, temp_dim_tid=args.temp_dim_tid, temp_dim_diw=args.temp_dim_diw)
    elif args.model_name == 'STGCN':
        model = STGCN(adj_mx=args.adj_mx, device=args.device, num_nodes=args.num_nodes, input_dim=args.input_dim,
                      output_dim=args.output_dim, window=args.window, horizon=args.horizon,
                      spatial_channels=args.spatial_channels, hidden_channel=args.hidden_channel)
    elif args.model_name == 'MOHER':
        model = MOHER(device=args.device, adj_mx=args.adj_mx, num_nodes=args.num_nodes, window=args.window,
                      horizon=args.horizon, input_dim=args.input_dim, output_dim=args.output_dim,
                      gamma=args.gamma, beta=args.beta, subgraph_size=args.subgraph_size, static_feat=args.node_fea,
                      n_heads=args.n_heads, n_layers=args.gcn_depth, hidden_dim=args.node_dim, dropout=args.dropout,
                      summarize=args.summarize, add_time=args.add_time)
    elif args.model_name == 'STCGNN':
        model = STCGNN(device=args.device, num_nodes=args.num_nodes, Ks=args.Ks, Kc=args.Kc, input_dim=args.input_dim,
                       output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layers=args.nn_layers,
                       in_window=args.window, out_horizon=args.horizon, As=args.As, Ac=args.Ac, add_time=args.add_time)
    elif args.model_name == 'STSHN':
        model = STSHN(device=args.device, input_dim=args.input_dim, output_dim=args.output_dim,
                      num_nodes=args.num_nodes, window=args.window, horizon=args.horizon,
                      spatial_layers=args.spatial_layers, temporal_layers=args.temporal_layers,
                      embed_dim=args.embed_dim, adj=args.adj, heads=args.heads, dropout=args.dropout,
                      hyper_num=args.hyper_num)
    elif args.model_name == 'CoGNN':
        model = CoGNN(gcn_true=args.gcn_true, buildA_true=args.buildA_true, gcn_depth=args.gcn_depth,
                      num_nodes=args.num_nodes, device=args.device, predefined_A=args.adj_mx,
                      dropout=args.dropout, subgraph_size=args.subgraph_size, node_dim=args.node_dim,
                      dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels,
                      residual_channels=args.residual_channels, skip_channels=args.skip_channels,
                      end_channels=args.end_channels, window=args.window, horizon=args.horizon, in_dim=args.input_dim,
                      out_dim=args.output_dim, layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha)
    elif args.model_name == 'STMAN':
        model = STMAN(input_dim=args.input_dim, P=args.window, Q=args.horizon, drop_rate=args.dropout)
    else:
        raise ValueError(f"Model {args.model_name} is not found")
    return model


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
