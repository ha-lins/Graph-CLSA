import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
from BYOL import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb

import random


def train(args, DS, gpu, num_gc_layers=4, epoch=40, batch=64):


    accuracies = {'val':[], 'test':[]}
    epochs = epoch
    log_interval = 10
    batch_size = batch
    # batch_size = 512
    lr = args.lr
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # print(path)
    dataset = TUDataset(path, name=DS, aug=args.aug, stro_aug=args.stro_aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none',
        stro_aug='none').shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    online_encoder = Encoder(dataset_num_features, args.hidden_dim, num_gc_layers)
    model = BYOL(online_encoder, args.hidden_dim, num_gc_layers,
        use_momentum=False).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.online_encoder.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(num_gc_layers))
    print('================')

    model.eval()
    emb, y = model.online_encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """

    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:

            # print('start')
            data, data_weak_aug, data_stro_aug = data
            optimizer.zero_grad()


            node_num, _ = data.x.size()
            data = data.to(device)
            # x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                edge_idx = data_weak_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_weak_aug.x = data_weak_aug.x[idx_not_missing]


                data_weak_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_weak_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            if args.stro_aug == 'stro_dnodes' or args.stro_aug == \
                    'stro_subgraph' or args.stro_aug \
                    == 'random2' or args.stro_aug == 'random3' or args.stro_aug == 'random4':
                edge_idx = data_stro_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_stro_aug.x = data_stro_aug.x[idx_not_missing]

                data_stro_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_stro_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_weak_aug = data_weak_aug.to(device)
            data_stro_aug = data_stro_aug.to(device)


            # weak_proj, x_proj = model(data.x, data.edge_index, data.batch,
            #     data.num_graphs, data_weak_aug.x, data_weak_aug.edge_index,
            #     data_weak_aug.batch, data_weak_aug.num_graphs)
            # target = model.loss_cal(x_proj, weak_proj)
            # loss_C = - torch.log(target).mean()
            #
            # stro_proj, x_proj = model(data.x, data.edge_index, data.batch,
            #     data.num_graphs, data_stro_aug.x, data_stro_aug.edge_index,
            #     data_stro_aug.batch, data_stro_aug.num_graphs)
            # prediction = model.loss_cal(x_proj, stro_proj)
            # loss_D = model.clsa_loss(prediction, target)

            # loss = loss_D.item() * data.num_graphs + loss_C
            # print('Loss {}, Loss_D {}, Loss_C'.format(loss, loss_D, loss_C))

            loss = model(data_weak_aug.x, data_weak_aug.edge_index,
                data_weak_aug.batch, data_weak_aug.num_graphs, data_stro_aug.x, data_stro_aug.edge_index,
                data_stro_aug.batch, data_stro_aug.num_graphs)
            loss_all += loss

            loss.backward()
            optimizer.step()
            # model.update_moving_average()

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.online_encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            # print(accuracies['val'][-1], accuracies['test'][-1])


    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_BYOL_' + args.DS + '_' + args.aug + '_' + args.stro_aug,
            'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},bs:{},epoch:{},layers:{},{},gpu:{},{},{},{},{},{}\n'.format(
            args.DS, batch, epoch, layers, tpe, gpu, num_gc_layers, epochs,
            log_interval, lr, s))
        f.write('\n')

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = arg_parse()

    gpu = 2
    #torch.cuda.set_device(gpu)
    layers = [3, 2, 8, 12]
    batch = [128, 256, 32, 16, 64]
    epoch = [20, 40, 100]
    ds = args.DS,#'IMDB-BINARY',#REDDIT-MULTI-5K', #['MUTAG'] ['MUTAG',
    # 'PTC_MR',
    # 'REDDIT-BINARY',
    # 'REDDIT-MULTI-5K']
    # seeds = [456, 465, 546, 564, 645, 654]
    seeds = [123, 132, 321, 312, 231]

    for d in ds:
        print(f'####################{d}####################')
        for l in layers:
            for b in batch:
                for e in epoch:
                    for i in range(5):
                        seed = seeds[i]
                        torch.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        np.random.seed(seed)
                        print(f'Dataset: {d}, Layer:{l}, Batch: {b}, Epoch: {e}, Seed: {seed}')
                        train(args, d, gpu, l, e, b)
        print('################################################')