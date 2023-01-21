import sys
import json
import types
import torch
import random
import numpy as np
from models import *


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

    with open(f"models/{model}/config.json", 'r') as f:
        model_cfg = json.load(f)

    cfg = types.SimpleNamespace()
    for c in [run_cfg, model_cfg, data_cfg]:
        for k, v in c.items():
            setattr(cfg, k, v)
    return cfg


def get_auxiliary(args, dataloader):
    """
    A simple example:
        if args.model_name == 'GWNet':
            df = h5py.File(os.path.join('./data/h5data', args.data + '.h5'), 'r')
            ret['adj_mx'] = np.array(df['adjacency_matrix'])
    """
    ret = {}
    # If your add models who require auxiliary information such as predefined graphs, write here
    pass
    return ret


def get_model(args):
    if args.model_name == 'SimMST':
        model = SimMST(device=args.device, num_nodes=args.num_nodes, gcn_depth=args.gcn_depth, dropout=args.dropout,
                       input_dim=args.input_dim, output_dim=args.output_dim, window=args.window, horizon=args.horizon,
                       subgraph_size=args.subgraph_size, node_dim=args.node_dim, tanhalpha=args.tanhalpha,
                       add_time=args.add_time, layers=args.layers, residual_channels=args.residual_channels,
                       conv_channels=args.conv_channels, skip_channels=args.skip_channels,
                       end_channels=args.end_channels, cross=args.cross, temporal_func=args.temporal_func)
    # initialize your models here
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
