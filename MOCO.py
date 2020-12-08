# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import copy

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=1024, m=0.999, T=0.07,
            mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        self.ratio = 1.0
        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.proj_head.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # print(ptr, self.queue[:, ptr:ptr + batch_size].size(), keys.T.size())
        if ptr + batch_size > self.K:
            ptr = 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, x, edge_index, batch, num_graphs, x_aug, edge_index_aug,
            batch_aug, num_graphs_aug, x_stro, edge_index_stro, batch_stro,
            num_graphs_stro):
        """
        Input:
            im_q: a batch of query graphs
            im_k: a batch of key graphs
        Output:
            logits, targets
        """

        # compute query features
        x_q, q = self.encoder_q(x, edge_index, batch)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(x_aug, edge_index_aug,
            #     batch_aug)

            x_k, k = self.encoder_k(x_aug, edge_index_aug, batch_aug) #keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # print('x_q:{}, x_k:{}, q:{}, k:{}'.format(x_q.size(), x_k.size(),
        #     q.size(), k.size()))
        l_pos = torch.einsum('nc,nc->n', [q[:k.shape[0], :], k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        # print(l_pos.shape, l_neg.shape)
        logits = torch.cat([l_pos, l_neg[:l_pos.shape[0],:]], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_contrastive = self.criterion(logits, labels)

        # compute ddm loss below
        # get P(Zk, Zi')
        p_weak = nn.functional.softmax(logits, dim=-1)

        loss_ddm = 0

        # for img_s in img_stronger_aug_list:
        x_s, q_s = self.encoder_q(x_stro, edge_index_stro, batch_stro)
        q_s = nn.functional.normalize(q_s, dim=1)
        # compute logits using the same set of code above
        # negative logits: NxK
        l_neg_stronger_aug = torch.einsum('nc,ck->nk', [q_s, self.queue.clone().detach()])
        # print(k.size(), q_s.size())
        l_pos_stronger_aug = torch.einsum('nc,nc->n', [q_s, k[:q_s.shape[0],:]]).unsqueeze(-1)

        # logits: Nx(1+K)
        logits_s = torch.cat([l_pos_stronger_aug, l_neg_stronger_aug], dim=1)
        logits_s /= self.T

        # compute nll loss below as -P(q, k) * log(P(q_s, k))
        log_p_s = nn.functional.log_softmax(logits_s, dim=-1)
        nll = -1.0 * torch.einsum('nk,nk->n', [p_weak[:log_p_s.shape[0]],
                                               log_p_s])
        loss_ddm = loss_ddm + torch.mean(nll) # average over the batch dimension

        loss = loss_contrastive + self.ratio * loss_ddm
        # print('loss_contrastive:{}, ddm:{}'.format(loss_contrastive, loss_ddm))
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
