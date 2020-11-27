import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

#from kornia import augmentation as augs
#from kornia import filters, color
# from utils import drop_adj, drop_feature
# from argparser import args
# helper functions
from arguments import arg_parse

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model): 
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            #TODO nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class CLSA(nn.Module):
    def __init__(self, net, hidden_dim, num_gc_layers, alpha=0.5, beta=1., 
            gamma=.1, projection_size = 256, projection_hidden_size = 4096, moving_average_decay = 0.99):
        super(CLSA, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        args = arg_parse()

        self.prior = args.prior

        self.embedding_dim = hidden_dim * num_gc_layers
        self.online_encoder = net

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, edge_index, batch, num_graphs, x_aug, edge_index_aug,
            batch_aug, num_graphs_aug):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        online_repr, online_proj = self.online_encoder(x_aug, edge_index_aug,
            batch_aug)
        # online_pred = self.online_predictor(online_proj)
        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_repr, target_proj = target_encoder(x, edge_index, batch)

        return online_proj, target_proj

    def loss_cal(self, x, x_aug):
        # print('[info] x: ({}, {}), x_aug: ({}, {})'.format(x, x.size(), x_aug,
        #     x_aug.size()))
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        for i in range(batch_size):
            for K in range():
                cond_dis = sim_matrix[]

        loss = pos_sim / sim_matrix.sum(dim=1)
        # loss = - torch.log(loss) #.mean()
        return loss

    def clsa_loss(self, prediction, target):

        prediction = torch.log(prediction)

        return -target.mul(prediction).mean()#sum() / target.shape[0]


# main class
# class CLSA(nn.Module):
#     def __init__(self, net, graph_size, hidden_layer = -2, projection_size = 256, projection_hidden_size = 4096, augment_fn = None, moving_average_decay = 0.99):
#         super().__init__()
#         # default SimCLR augmentation
#         '''
#         DEFAULT_AUG = nn.Sequential(
#             RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
#             augs.RandomGrayscale(p=0.2),
#             augs.RandomHorizontalFlip(),
#             RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
#             augs.RandomResizedCrop((image_size, image_size)),
#             color.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
#         )
#         '''
# 
#         #self.augment = default(augment_fn, DEFAULT_AUG)
#         #self.augment = graph_aug(augment_fn, DEFAULT_AUG)
# 
#         self.online_encoder = net #NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
#         self.target_encoder = None
#         self.target_ema_updater = EMA(moving_average_decay)
# 
#         self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
# 
#         # send a mock image tensor to instantiate singleton parameters
#         #self.forward(torch.randn(2, 3, image_size, image_size))
# 
#     # adj - [B,N,N], feat - [B,N,feat_dim]
#     def graph_aug(self, adj, feat):
#         return drop_adj(adj, args.drop_prob), feat #drop_feature(feat,
#         # args.drop_prob)
# 
#     @singleton('target_encoder')
#     def _get_target_encoder(self):
#         target_encoder = copy.deepcopy(self.online_encoder)
#         return target_encoder
# 
#     def reset_moving_average(self):
#         del self.target_encoder
#         self.target_encoder = None
# 
#     def update_moving_average(self):
#         assert self.target_encoder is not None, 'target encoder has not been created yet'
#         update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
# 
#     def forward(self, adj, feat, diff=None, mask=None):
#         adj_one, feat_one = self.graph_aug(adj,feat)
#         adj_two, feat_two = self.graph_aug(adj,feat) #self.augment(x), self.augment(x)
# 
#         online_proj_one, _ = self.online_encoder(adj_one, feat_one)
#         online_proj_two, _ = self.online_encoder(adj_two, feat_two)
# 
#         online_pred_one = self.online_predictor(online_proj_one)
#         online_pred_two = self.online_predictor(online_proj_two)
# 
#         with torch.no_grad():
#             target_encoder = self._get_target_encoder()
#             target_proj_one, _ = target_encoder(adj_one, feat_one)
#             target_proj_two, _ = target_encoder(adj_two, feat_two)
# 
#         loss_one = loss_fn(online_pred_one, target_proj_two.detach())
#         loss_two = loss_fn(online_pred_two, target_proj_one.detach())
# 
#         loss = loss_one + loss_two
#         return loss.mean()
# 
#     def embed(self, adj, diff, feat, mask=None):
#         online_l_one, _ = self.online_encoder(adj, feat)
#         online_l_two, _ = self.online_encoder(diff, feat)
# 
#         online_proj_one = online_l_one.sum(1)
#         online_proj_two = online_l_two.sum(1)
# 
#         return (online_proj_one + online_proj_two).detach()
