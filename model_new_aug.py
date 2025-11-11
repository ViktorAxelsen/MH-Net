import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.conv import SAGEConv
import random
import math
import dgl.nn.pytorch as dglnn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import SAGEConv
import time
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

class GCN_Hetro(nn.Module):
    def __init__(self, embedding_size, h_feats, dropout):
        super(GCN_Hetro, self).__init__()
        self.gcn_out_dim = 4 * h_feats
        self.h_embedding = nn.Embedding(256 + 1, embedding_size)
        self.p_embedding = nn.Embedding(256 + 1, embedding_size)
        self.h_p_embedding = nn.Embedding(256 + 1, embedding_size)
        self.gcn1 = dglnn.HeteroGraphConv({
            'h' : SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'p' : SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'h_p' : SAGEConv(embedding_size, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
            },aggregate='mean')
        self.gcn2 = dglnn.HeteroGraphConv({
            'h' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'p' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'h_p' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
            },aggregate='mean')
        self.gcn3 = dglnn.HeteroGraphConv({
            'h' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'p' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'h_p' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
            },aggregate='mean')
        self.gcn4 = dglnn.HeteroGraphConv({
            'h' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'p' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats)),
            'h_p' : SAGEConv(h_feats, h_feats, 'mean', feat_drop=dropout, activation=nn.PReLU(h_feats), norm=nn.BatchNorm1d(h_feats))
            },aggregate='mean')
            
    def forward(self, g):
        h = g.nodes['header'].data['feat'].long()
        p = g.nodes['payload'].data['feat'].long()
        h_p = g.nodes['header_p'].data['feat'].long()
        src_h = h.view(-1)
        src_p = p.view(-1)
        src_hp = h_p.view(-1)
        emd_h = self.h_embedding(src_h)
        emd_p = self.p_embedding(src_p)
        emd_hp = self.h_p_embedding(src_hp)
        h = {
            'header' : emd_h,
            'payload' : emd_p,
            'header_p': emd_hp,
        }
        h1 = self.gcn1(g, h)
        h2 = self.gcn2(g, h1) 
        h3 = self.gcn3(g, h2)
        h4 = self.gcn4(g, h3)
        
        g.nodes['header'].data['h'] = torch.cat((h1['header'], h2['header'], h3['header'], h4['header']), dim=1)  
        g.nodes['payload'].data['h'] = torch.cat((h1['payload'], h2['payload'], h3['payload'], h4['payload']), dim=1) 
        g.nodes['header_p'].data['h'] = torch.cat((h1['header_p'], h2['header_p'], h3['header_p'], h4['header_p']), dim=1)  
        g_vec_h = dgl.mean_nodes(g, 'h',ntype='header')
        g_vec_p = dgl.mean_nodes(g,'h',ntype='payload')
        g_vec_hp = dgl.mean_nodes(g,'h',ntype='header_p')
        g_vec = torch.cat((g_vec_h,g_vec_p,g_vec_hp),dim = 1)
        return g_vec

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MixTemporalGNN(nn.Module):
    def __init__(self, num_classes, embedding_size=64, h_feats=128, dropout=0.2, downstream_dropout=0.0,
                 point=15, seq_aug_ratio=0.8, drop_edge_ratio=0.1, drop_node_ratio=0.1, K=15,
                 hp_ratio=0.5, tau=0.07, gtau=0.07):
        super(MixTemporalGNN, self).__init__()
        self.hetro_graphConv = GCN_Hetro(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        self.bit_hetro_graphConv = GCN_Hetro(embedding_size=embedding_size, h_feats=h_feats, dropout=dropout)
        
        self.hidden_size = embedding_size // 2
        self.gcn_out_dim = 4 * h_feats
        self.drop_edge_rate = 0
        self.point = point

        self.seq_aug_ratio = seq_aug_ratio
        self.drop_edge_ratio = drop_edge_ratio
        self.drop_node_ratio = drop_node_ratio
        self.K = K
        self.restart_prob = 0.8
        self.rw_hop = 32
        self.hp_ratio = hp_ratio

        self.rnn = nn.LSTM(input_size=self.gcn_out_dim * 3, hidden_size=self.gcn_out_dim * 3, num_layers=2, bidirectional=True, dropout=downstream_dropout)
        self.bit_rnn = nn.LSTM(input_size=self.gcn_out_dim * 3, hidden_size=self.gcn_out_dim * 3, num_layers=2, bidirectional=True, dropout=downstream_dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 6, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim)
        )
        
        self.bit_fc = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 6, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim)
        )
        
        self.cls = nn.Linear(in_features=self.gcn_out_dim * 2, out_features=num_classes)
        self.bit_cls = nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)
        
        self.packet_head = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 6, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim),
            nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)
        )
        self.bit_packet_head = nn.Sequential(
            nn.Linear(in_features=self.gcn_out_dim * 4, out_features=self.gcn_out_dim),
            nn.PReLU(self.gcn_out_dim),
            nn.Linear(in_features=self.gcn_out_dim, out_features=num_classes)
        )
        
        self.mask_rate = 0.3
        self.supcl = SupConLoss(temperature=tau, base_temperature=tau)
        self.supcl_g = SupConLoss(temperature=gtau, base_temperature=gtau)
        self.drop_edge_trans = dgl.DropEdge(p=self.drop_edge_ratio)
        self.drop_node_trans = dgl.DropNode(p=self.drop_node_ratio)

    @timer_decorator
    def forward(self, hetro_data,bit_hetro_data,labels,hetro_mask,bit_hetro_mask):
        graph_cl_loss,hetro_gcn_out,hetro_mask,aug_hetro_gcn_out \
            = self.feat_graph_cl_loss(hetro_data,labels,hetro_mask,self.hetro_graphConv)
        bit_graph_cl_loss,bit_hetro_gcn_out,bit_hetro_mask,bit_aug_hetro_gcn_out \
            = self.feat_graph_cl_loss(bit_hetro_data,labels,bit_hetro_mask,self.bit_hetro_graphConv)
        
        print("augment finished")
        
        gcn_out = hetro_gcn_out
        bit_gcn_out = bit_hetro_gcn_out
        print("packet-level training")
        # packet-level head
        packet_mask = hetro_mask
        packet_rep = gcn_out[:, :self.K, :].reshape(-1, gcn_out.shape[2])[packet_mask]
        packet_label = labels.reshape(-1, 1).repeat(1, self.point)[:, :self.K].reshape(-1)[packet_mask]
        bit_packet_rep = bit_gcn_out[:, :self.K, :].reshape(-1, bit_gcn_out.shape[2])[packet_mask]
        
        res_packet_rep = torch.cat((packet_rep,bit_packet_rep),dim=-1)
        packet_out = self.packet_head(res_packet_rep)  
        print("packet-level training finished")
        
        gcn_out_aug = aug_hetro_gcn_out
        bit_gcn_out_aug = bit_aug_hetro_gcn_out
        
        aug_index = []
        bit_aug_index = []
        for _ in range(len(gcn_out_aug)):
            index = np.random.choice(range(self.point), size=int(self.point * self.seq_aug_ratio), replace=False)
            index.sort()
            aug_index.append(index)

        for _ in range(len(bit_gcn_out_aug)):
            index = np.random.choice(range(self.point), size=int(self.point * self.seq_aug_ratio), replace=False)
            index.sort()
            bit_aug_index.append(index)
        
        aug_index = torch.tensor(np.array(aug_index), dtype=int, device=gcn_out.device)
        aug_index = aug_index.unsqueeze(2)
        aug_index = aug_index.repeat(1, 1, gcn_out_aug.shape[2])
        
        bit_aug_index = torch.tensor(np.array(bit_aug_index), dtype=int, device=gcn_out.device)
        bit_aug_index = bit_aug_index.unsqueeze(2)
        bit_aug_index = bit_aug_index.repeat(1, 1, bit_gcn_out_aug.shape[2])

        gcn_out_aug = torch.gather(gcn_out_aug, dim=1, index=aug_index)
        bit_gcn_out_aug = torch.gather(bit_gcn_out_aug,dim=1,index = bit_aug_index)

        gcn_out = gcn_out.transpose(0, 1)
        _, (h_n, _) = self.rnn(gcn_out)
        rnn_out = torch.cat((h_n[-1], h_n[-2]), dim=1)
        bit_gcn_out = bit_gcn_out.transpose(0,1)
        _, (bit_h_n, _) = self.bit_rnn(bit_gcn_out)
        bit_rnn_out = torch.cat((bit_h_n[-1], bit_h_n[-2]), dim=1)

        gcn_out_aug = gcn_out_aug.transpose(0, 1)
        _, (h_n_aug, _) = self.rnn(gcn_out_aug)
        rnn_out_aug = torch.cat((h_n_aug[-1], h_n_aug[-2]), dim=1)
        bit_gcn_out_aug = bit_gcn_out_aug.transpose(0, 1)
        _, (bit_h_n_aug, _) = self.bit_rnn(bit_gcn_out_aug)
        bit_rnn_out_aug = torch.cat((bit_h_n_aug[-1], bit_h_n_aug[-2]), dim=1)
        
        rnn_out = F.normalize(rnn_out, p=2)
        rnn_out_aug = F.normalize(rnn_out_aug, p=2)
        bit_rnn_out = F.normalize(bit_rnn_out, p=2)
        bit_rnn_out_aug = F.normalize(bit_rnn_out_aug, p=2)
        
        cl_loss = self.supcl(torch.cat((rnn_out.unsqueeze(1), rnn_out_aug.unsqueeze(1)), dim=1), labels)
        bit_cl_loss = self.supcl(torch.cat((bit_rnn_out.unsqueeze(1), bit_rnn_out_aug.unsqueeze(1)), dim=1), labels)
        
        out = self.fc(rnn_out)
        bit_out = self.bit_fc(bit_rnn_out)
        
        out = torch.cat((out,bit_out),dim = -1)
        
        out = self.cls(out)
        return out, cl_loss, graph_cl_loss, packet_out, packet_label,bit_graph_cl_loss,bit_cl_loss
    
    @timer_decorator
    def feat_graph_cl_loss(self,hetro_data,labels,hetro_mask,hetro_graphConv):
        hetro_mask = hetro_mask.reshape(labels.shape[0], self.point, -1)[:, :self.K, :].reshape(-1)

        aug_hetro_graph_data = self.constrative_samples_generation(hetro_data)
        aug_feat_hetro_graph_data = self.feat_constrative_samples_generation(hetro_data)
        
        hetro_gcn_out = hetro_graphConv(hetro_data).reshape(
            labels.shape[0], self.point, -1)
        subgraph_aug_hetro_gcn_out = hetro_graphConv(aug_hetro_graph_data).reshape(
            labels.shape[0], self.point, -1)
        feat_aug_hetro_gcn_out = hetro_graphConv(aug_feat_hetro_graph_data).reshape(
            labels.shape[0], self.point, -1)
        
        temp_src = hetro_gcn_out[:, :self.K, :].reshape(-1, hetro_gcn_out.shape[2])[hetro_mask]
        temp_subgraph = subgraph_aug_hetro_gcn_out[:, :self.K, :].reshape(-1, subgraph_aug_hetro_gcn_out.shape[2])[hetro_mask]
        temp_feat = feat_aug_hetro_gcn_out[:, :self.K, :].reshape(-1, feat_aug_hetro_gcn_out.shape[2])[hetro_mask]
        mask_src_subgraph = torch.any(temp_src != 0, dim=1) & torch.any(temp_subgraph != 0, dim=1)
        mask_src_feat = torch.any(temp_src != 0,dim = 1) & torch.any(temp_feat != 0,dim = 1)

        hetro_label = labels.reshape(-1, 1).repeat(1, self.point)[:, :self.K].reshape(-1)[hetro_mask]

        subgraph_hetro_cl_loss = self.supcl_g(torch.cat((F.normalize(temp_src[mask_src_subgraph], p=2).unsqueeze(1), F.normalize(temp_subgraph[mask_src_subgraph], p=2).unsqueeze(1)), dim=1), hetro_label[mask_src_subgraph])
        feat_hetro_cl_loss = self.supcl_g(torch.cat((F.normalize(temp_src[mask_src_feat], p=2).unsqueeze(1), F.normalize(temp_feat[mask_src_feat], p=2).unsqueeze(1)), dim=1), hetro_label[mask_src_feat])
        
        graph_cl_loss =  feat_hetro_cl_loss + subgraph_hetro_cl_loss

        return graph_cl_loss,hetro_gcn_out,hetro_mask,subgraph_aug_hetro_gcn_out    
    
    @timer_decorator
    def feat_constrative_samples_generation(self,header_graph_data):
        aug_feat_hetro_graph_data = []
        for i,g in enumerate(dgl.unbatch(header_graph_data)):
            feat = g.nodes['header'].data['feat']
            g.nodes['header'].data['feat'] = torch.flip(feat,dims=[0])
            feat = g.nodes['payload'].data['feat']
            g.nodes['payload'].data['feat'] = torch.flip(feat,dims=[0])
            feat = g.nodes['header_p'].data['feat']
            g.nodes['header_p'].data['feat'] = torch.flip(feat,dims=[0])
            aug_feat_hetro_graph_data.append(g)
        aug_feat_hetro_graph_data = dgl.batch(aug_feat_hetro_graph_data)
        return  aug_feat_hetro_graph_data
    
    def rwr_trace_to_dgl_graph(self,g,seed,header_trace,payload_trace,header_p_trace):
        header_subv = torch.unique(header_trace).tolist()
        payload_subv = torch.unique(payload_trace).tolist()
        header_p_subv = torch.unique(header_p_trace).tolist()
        
        subg = dgl.node_subgraph(g,{'header':header_subv,'payload':payload_subv,'header_p':header_p_subv},store_ids=False)
        
        return subg
    
    def max_node(self,g,idx,edge_type,max_node):
        max_nodes_per_seed = min(
                25,
                int(
                    (
                        (g.in_degrees(idx,etype = edge_type) ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                    )
                    + 0.5
                ),
            )
        return max_nodes_per_seed    
    
    @timer_decorator
    def constrative_samples_generation(self,hetro_graph_data):
        aug_hetro_graph_data = []
        
        for i,g in enumerate(dgl.unbatch(hetro_graph_data)):
            if g.num_nodes('header')  == 0 or g.num_nodes('payload') == 0 or g.num_nodes('header_p') == 0:
                aug_hetro_graph_data.append(g)
                continue
            idx_header = random.sample(list(g.nodes('header')),1)
            idx_payload = random.sample(list(g.nodes('payload')),1)
            idx_header_p = random.sample(list(g.nodes('header_p')),1)
            try:
                header_node_length = self.max_node(g,idx_header,'h',g.num_nodes('header') // 2)
                header_traces = dgl.sampling.random_walk(g,idx_header,metapath = ['h'] * header_node_length)
                payload_node_length = self.max_node(g,idx_payload,'p',g.num_nodes('payload') // 2)
                payload_traces = dgl.sampling.random_walk(g,idx_payload,metapath = ['p'] * payload_node_length)
                header_p_node_length = self.max_node(g,idx_header_p,'h_p',g.num_nodes('header_p') // 2)
                header_p_traces = dgl.sampling.random_walk(g,idx_header_p,metapath = ['h_p'] * header_p_node_length)
            except:
                print('random_walk error')
                print(g.num_nodes('header'),g.num_nodes('payload'),g.num_nodes('header_p'))
                aug_hetro_graph_data.append(g)
                continue
            aug_hetro_graph_data.append(self.rwr_trace_to_dgl_graph(g,idx_header,header_traces[0][0],payload_traces[0][0],header_p_traces[0][0]))
        aug_hetro_graph_data = np.array(aug_hetro_graph_data).flatten()
        aug_hetro_graph_data = dgl.batch(aug_hetro_graph_data)
        return aug_hetro_graph_data

if __name__ == '__main__':
    pass

