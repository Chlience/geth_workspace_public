import ragdoll
from ragdoll.perf_counter import perf_counter
from dgl.data import load_data
from ragdoll.data.data_wrapper import DataWrapper
import networkx as nx
import numpy as np
import scipy
import dgl
from dgl import DGLGraph
import torch


class Dataset(object):
    def __init__(self, is_root, args):
        self.name = args.dataset
        self.is_root = is_root
        self._load(args)

    def pad_to_128(self, a):
        assert len(a.shape) == 2
        d = a.shape[1]
        padding = (128 - d % 128) % 128
        if padding == 0:
            return
        return np.pad(a, ((0, 0), (0, padding)), constant_values=0)

    def _load(self, args):
        data = None
        n_nodes = None
        perf_counter.record_start("load_dgl_graph")
        if self.is_root:
            #dgl_graph = load_data(args)[0]
            if args.dataset != "cora":                
                dgl_graph, label_dict = dgl.load_graphs("/workspace/graph/graphdata/enwiki-2013.mtx_dgl_graph.bin")
                dgl_graph = dgl_graph[0]

                mygraph=dgl_graph
                n_feats=args.feat_size
                n_classes=8
                f_tensor=torch.randn(mygraph.num_nodes(),n_feats)
                l_tensor=torch.randint(0,n_classes - 1,(mygraph.num_nodes(),)).type(torch.int64)
                testmask=torch.ones(mygraph.num_nodes()).type(torch.bool)
                trainmask=torch.ones(mygraph.num_nodes()).type(torch.bool)
                mygraph.ndata['label'] = l_tensor
                mygraph.ndata['feat'] = f_tensor
                mygraph.ndata['val_mask']=testmask
                mygraph.ndata['test_mask']=testmask
                mygraph.ndata['train_mask']=trainmask
            else:
                dgl_graph = load_data(args)[0]

            n_nodes = dgl_graph.number_of_nodes()
            
            graph = dgl_graph.adj_external(scipy_fmt="csr")

            data = dgl_graph
            data.graph = graph
        
        perf_counter.record_end("load_dgl_graph")

        # print('n classes:', data.num_labels)
        # print('feat size', data.features.shape[-1])
        data = DataWrapper(data)
        n_nodes = DataWrapper(n_nodes)

        g = data.get_attr('graph')
        indptr = g.get_attr('indptr')
        indices = g.get_attr('indices')

        perf_counter.record_start("partition_graph")
        sg_n, sg_xadj, sg_adjncy = ragdoll.partition_graph(
            n_nodes.get_val(0), indptr.get_val([]), indices.get_val([]))
        perf_counter.record_end("partition_graph")

        perf_counter.record_start("check_and_build_csr")
        sg_e = sg_xadj[sg_n]
        assert sg_e == len(sg_adjncy)
        assert sg_n + 1 == len(sg_xadj)
        for u in sg_adjncy:
            assert u >= 0 and u < sg_n
        for i in range(sg_n):
            assert sg_xadj[i + 1] >= sg_xadj[i]

        edge_data = np.ones([sg_e])
        subgraph = scipy.sparse.csr_matrix(
            (edge_data, sg_adjncy, sg_xadj), shape=[sg_n, sg_n])
        perf_counter.record_end("check_and_build_csr")

        self.local_n_nodes = ragdoll.get_local_n_nodes()
        self.n_nodes = sg_n

        #print('To nx sparse matrix')

        #self.graph = nx.from_scipy_sparse_matrix(
        #    subgraph, create_using=nx.DiGraph())
        #print('To nx sparse matrix Done')
        self.graph = subgraph
        features = data.get_attr('ndata').get_item('feat')
        # print(dir(data.get_attr('ndata').get_item('label').data))
        labels = data.get_attr('ndata').get_item('label').call_func('type', torch.int32)
        train_mask = data.get_attr('ndata').get_item('train_mask').call_func('type', torch.int32)
        val_mask = data.get_attr('ndata').get_item('val_mask').call_func('type', torch.int32)
        test_mask = data.get_attr('ndata').get_item('test_mask').call_func('type', torch.int32)

        if self.is_root:
            print('feature shape is', features.get_val().shape)
            print('labels shape is', labels.get_val().shape)
            print('train mask shape is', train_mask.get_val().shape)

        perf_counter.record_start("dispatch_feat")
        features = ragdoll.dispatch_float(
            features.get_val(), args.feat_size, sg_n, no_remote=1)[:self.local_n_nodes*args.feat_size]
        self.features = np.reshape(features, [-1, args.feat_size])
        self.features = self.pad_to_128(self.features)
        args.feat_size = self.features.shape[-1]
        perf_counter.record_end("dispatch_feat")


        perf_counter.record_start("dispatch_labels_others")
        labels = ragdoll.dispatch_int(labels.get_val(), 1, sg_n, no_remote=1)[
            :self.local_n_nodes]
        
        train_mask = ragdoll.dispatch_int(
            train_mask.get_val(), 1, sg_n, no_remote=1)[:self.local_n_nodes]
        val_mask = ragdoll.dispatch_int(
            val_mask.get_val(), 1, sg_n, no_remote=1)[:self.local_n_nodes]
        test_mask = ragdoll.dispatch_int(
            test_mask.get_val(), 1, sg_n, no_remote=1)[:self.local_n_nodes]
        self.labels = np.reshape(labels, [-1])
        self.train_mask = np.reshape(train_mask, [-1])
        self.val_mask = np.reshape(val_mask, [-1])
        self.test_mask = np.reshape(test_mask, [-1])

        perf_counter.record_end("dispatch_labels_others")

        print('My feature shape is', self.features.shape)
        print('My labels shape is', self.labels.shape)
        print('My train mask shape is', self.train_mask.shape)
        print('My val mask shape is', self.val_mask.shape)
        print('My test mask shape is', self.test_mask.shape)

        g = self.graph

        #if hasattr(args, 'self_loop'):
        #    if args.self_loop:
        #        g.remove_edges_from(nx.selfloop_edges(g))
        #        g.add_edges_from(zip(g.nodes(), g.nodes()))
        perf_counter.record_start("build_graph_to_cuda")
        g = DGLGraph(g)
        self.graph = g.to("cuda")
        perf_counter.record_end("build_graph_to_cuda")

