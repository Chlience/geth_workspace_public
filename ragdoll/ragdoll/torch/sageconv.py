"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn
from torch.nn import functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair, check_eq_shape

from . import GraphAllgatherFunc, GraphAllgatherOutplaceFunc


class SAGEConv(nn.Module):
    r"""GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 n_nodes,
                 local_n_nodes,
                 apply_gather=False,
                 no_remote=True,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._n_nodes = n_nodes
        self._local_n_nodes = local_n_nodes
        self._no_remote = no_remote
        self._apply_gather = apply_gather
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        
    def apply_allgather(self, t):
        if not self._apply_gather:
            return t
        if self._no_remote:
            assert t.shape[0] == self._local_n_nodes
            t = GraphAllgatherOutplaceFunc.apply(
                t, self._n_nodes, self._local_n_nodes)
        else:
            t = GraphAllgatherFunc.apply(t)
        return t
    
    def slice_local(self, t):
        assert t.shape[0] == self._n_nodes
        t = t[:self._local_n_nodes, ...]
        return t

    def _lstm_reducer_minibatch(self, nodes):
        m = nodes.mailbox['m'] 
    
        mini_batch_size = 16384 
        results = []
    
        for i in range(0, m.shape[0], mini_batch_size):
            end_idx = min(i + mini_batch_size, m.shape[0])
            mini_m = m[i:end_idx]
            mini_batch = mini_m.shape[0]
        
            h = (mini_m.new_zeros((1, mini_batch, self._in_src_feats)),
                 mini_m.new_zeros((1, mini_batch, self._in_src_feats)))
            _, (rst, _) = self.lstm(mini_m, h)
            results.append(rst.squeeze(0))
    
        final_rst = torch.cat(results, dim=0)
        return {'neigh': final_rst}

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        #print("========================into lstm_reducer",flush=True)
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        #print("=======================sageconv m_shape", m.shape)
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        # only support mean!
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)

            feat_src = self.apply_allgather(feat_src)
            feat_dst = feat_src[:self._local_n_nodes]
            h_self = feat_dst

            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                neigh = graph.dstdata['neigh']
                neigh = self.slice_local(neigh)
                h_neigh = neigh
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = graph.srcdata['h']    # same as above if homogeneous
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
                # divide in_degrees
                degree = graph.in_degrees()
                degree = degree[:self._local_n_nodes]
                degs = degree.to(feat_dst)
                print(self._local_n_nodes)
                neigh = self.slice_local(graph.dstdata['neigh'])
                h_neigh = (neigh) / (degs.unsqueeze(-1) + 1)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
                neigh = graph.dstdata['neigh']
                neigh = self.slice_local(neigh)
                h_neigh = neigh 
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                #print("==================verify print",flush=True)
                graph.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                neigh = graph.dstdata['neigh']
                neigh = self.slice_local(neigh)
                h_neigh = neigh 
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = self.fc_neigh(h_neigh)
            else:
                rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst

