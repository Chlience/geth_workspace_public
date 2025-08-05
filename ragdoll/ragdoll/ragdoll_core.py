import ctypes
import pathlib
import sysconfig
import time
import numpy as np
import torch

from ragdoll.perf_counter import perf_counter


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


class RagdollCore(object):
    def __init__(self):
        lib_name = "ragdoll_torch_ops" + get_ext_suffix()
        so_path = pathlib.Path(__file__).with_name(lib_name)
        # so_path = str(pathlib.Path(__file__).with_name('libragdoll_core.so'))
        print('so path is ', so_path)
        self.lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    def hello(self):
        self.lib.ragdoll_hello()

    def set_comm_pattern(self, comm):
        from . import torch
        torch.set_comm_pattern(comm)

    def init_logs(self, log_file):
        log_file = log_file.encode('utf-8')
        self.lib.ragdoll_init_logs(log_file)
    
    def deinit_logs(self):
        self.lib.ragdoll_deinit_logs()

    def init(self, device_id=-1):
        self.lib.ragdoll_init(device_id)
        
    def deinit(self):
        self.lib.ragdoll_deinit()

    def rank(self):
        return self.lib.ragdoll_rank()

    def device_id(self):
        return self.lib.ragdoll_device_id()

    def world_size(self):
        return self.lib.ragdoll_world_size()

    def partition_graph(self, n_nodes, xadj, adjncy, mini_graph_info=None):
        if isinstance(xadj, list):
            c_xadj = (ctypes.c_int * len(xadj))(*xadj)
            c_adjncy = (ctypes.c_int * len(adjncy))(*adjncy)
        else:
            c_xadj = xadj.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_adjncy = adjncy.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        sg_n = ctypes.c_int32()
        sg_xadj = ctypes.POINTER(ctypes.c_int)()
        sg_adjncy = ctypes.POINTER(ctypes.c_int)()
        
        if mini_graph_info is not None:
            mini_n = mini_graph_info['n']
            mini_xadj = mini_graph_info['xadj']
            mini_adjncy = mini_graph_info['adjncy']
            mini_gid2mid = mini_graph_info['gid2mid']
            mini_node_weights = mini_graph_info['node_weights']
            mini_edge_weights = mini_graph_info['edge_weights']
            # c_mini_xadj = (ctypes.c_int * len(mini_xadj))(*mini_xadj)
            # c_mini_adjncy = (ctypes.c_int * len(mini_adjncy))(*mini_adjncy)
            # c_mini_gid2mid = (ctypes.c_int * len(mini_gid2mid))(*mini_gid2mid)
            # c_mini_node_weights = (ctypes.c_int * len(mini_node_weights))(*mini_node_weights)
            # c_mini_edge_weights = (ctypes.c_int * len(mini_edge_weights))(*mini_edge_weights)
            c_mini_xadj = mini_xadj.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_mini_adjncy = mini_adjncy.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_mini_gid2mid = mini_gid2mid.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_mini_node_weights = mini_node_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_mini_edge_weights = mini_edge_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            mini_n = 0
            c_mini_xadj = ctypes.POINTER(ctypes.c_int)()
            c_mini_adjncy = ctypes.POINTER(ctypes.c_int)()
            c_mini_gid2mid = ctypes.POINTER(ctypes.c_int)()
            c_mini_node_weights = ctypes.POINTER(ctypes.c_float)()
            c_mini_edge_weights = ctypes.POINTER(ctypes.c_float)()
        
        self.lib.ragdoll_partition_graph(n_nodes, c_xadj, c_adjncy, ctypes.byref(
            sg_n), ctypes.byref(sg_xadj), ctypes.byref(sg_adjncy), mini_n, c_mini_xadj, c_mini_adjncy, c_mini_gid2mid, c_mini_node_weights, c_mini_edge_weights)

        n_edges = sg_xadj[sg_n.value]
        print('Subgraph nodes:', sg_n.value, 'local nodes',
              self.lib.ragdoll_get_local_n_nodes(), 'edges:', n_edges)
        
        py_sg_xadj = np.ctypeslib.as_array(sg_xadj, shape=(sg_n.value + 1,)).copy()
        py_sg_adjncy = np.ctypeslib.as_array(sg_adjncy, shape=(n_edges,)).copy()
        
        print("Finished converting to numpy")
        
        self.lib.ragdoll_release(sg_xadj)
        self.lib.ragdoll_release(sg_adjncy)
        return sg_n.value, py_sg_xadj, py_sg_adjncy
    
    def pre_partition_graph(self, n_peers, n_nodes, xadj, adjncy, mini_graph_info=None):
        start = time.time()
        if isinstance(xadj, list):
            c_xadj = (ctypes.c_int * len(xadj))(*xadj)
            c_adjncy = (ctypes.c_int * len(adjncy))(*adjncy)
        else:
            c_xadj = xadj.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_adjncy = adjncy.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        if mini_graph_info is not None:
            mini_n = mini_graph_info['n']
            mini_xadj = mini_graph_info['xadj']
            mini_adjncy = mini_graph_info['adjncy']
            mini_gid2mid = mini_graph_info['gid2mid']
            mini_node_weights = mini_graph_info['node_weights']
            mini_edge_weights = mini_graph_info['edge_weights']
            c_mini_xadj = mini_xadj.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_mini_adjncy = mini_adjncy.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_mini_gid2mid = mini_gid2mid.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            c_mini_node_weights = mini_node_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_mini_edge_weights = mini_edge_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            mini_n = 0
            c_mini_xadj = ctypes.POINTER(ctypes.c_int)()
            c_mini_adjncy = ctypes.POINTER(ctypes.c_int)()
            c_mini_gid2mid = ctypes.POINTER(ctypes.c_int)()
            c_mini_node_weights = ctypes.POINTER(ctypes.c_float)()
            c_mini_edge_weights = ctypes.POINTER(ctypes.c_float)()

        prepart_info_bin_size_c = ctypes.c_size_t()
        prepart_info_bin_c = ctypes.POINTER(ctypes.c_char)()
        self.lib.ragdoll_pre_partition_graph(n_peers, n_nodes, c_xadj, c_adjncy, mini_n, c_mini_xadj, c_mini_adjncy, c_mini_gid2mid, c_mini_node_weights, c_mini_edge_weights, ctypes.byref(prepart_info_bin_size_c), ctypes.byref(prepart_info_bin_c))
        
        prepart_info_bin = np.ctypeslib.as_array(prepart_info_bin_c, shape=(prepart_info_bin_size_c.value,)).copy()
        
        self.lib.ragdoll_release(prepart_info_bin_c)
        end = time.time()
        print("Prepartition overall took: {}s".format(end - start))
        return prepart_info_bin

    # todo: check multithread 250408
    def load_prepart_info(self, prepart_info_bin=None):
        sg_n = ctypes.c_int32()
        sg_xadj = ctypes.POINTER(ctypes.c_int)()
        sg_adjncy = ctypes.POINTER(ctypes.c_int)()
        
        prepart_info_bin_c = ctypes.POINTER(ctypes.c_char)()
        prepart_info_bin_size_c = ctypes.c_size_t()
        if prepart_info_bin is not None:
            prepart_info_bin_c = prepart_info_bin.ctypes.data_as(ctypes.POINTER(ctypes.c_char))
            prepart_info_bin_size_c = ctypes.c_size_t(len(prepart_info_bin))
            
        self.lib.ragdoll_load_prepart_info(
            ctypes.byref(sg_n), ctypes.byref(sg_xadj), ctypes.byref(sg_adjncy),
            prepart_info_bin_size_c, prepart_info_bin_c)

        n_edges = sg_xadj[sg_n.value]
        print('Subgraph nodes:', sg_n.value, 'local nodes',
              self.lib.ragdoll_get_local_n_nodes(), 'edges:', n_edges)
        
        py_sg_xadj = np.ctypeslib.as_array(sg_xadj, shape=(sg_n.value + 1,)).copy()
        py_sg_adjncy = np.ctypeslib.as_array(sg_adjncy, shape=(n_edges,)).copy()
        
        print("Finished converting to numpy")
        
        self.lib.ragdoll_release(sg_xadj)
        self.lib.ragdoll_release(sg_adjncy)
        return sg_n.value, py_sg_xadj, py_sg_adjncy

    def partition_graph_on_dir(self, dirname):
        sg_n = ctypes.c_int32()
        sg_xadj = ctypes.POINTER(ctypes.c_int)()
        sg_adjncy = ctypes.POINTER(ctypes.c_int)()
        dirname = dirname.encode('utf-8')
        self.lib.ragdoll_partition_graph_on_dir(dirname, ctypes.byref(
            sg_n), ctypes.byref(sg_xadj), ctypes.byref(sg_adjncy))

        n_edges = sg_xadj[sg_n.value]
        print('Subgraph nodes:', sg_n.value, 'local nodes',
              self.lib.ragdoll_get_local_n_nodes(), 'edges:', n_edges)
        py_sg_xadj = [sg_xadj[i] for i in range(sg_n.value + 1)]
        py_sg_adjncy = [sg_adjncy[i] for i in range(n_edges)]
        self.lib.ragdoll_release(sg_xadj)
        self.lib.ragdoll_release(sg_adjncy)
        return sg_n.value, py_sg_xadj, py_sg_adjncy
    
    def graph_detailed_info(self, n_nodes, world_size):
        gid2pid_c = ctypes.POINTER(ctypes.c_int)()
        num_local_nodes_c = ctypes.POINTER(ctypes.c_int)()
        gid2lid_unordered_c = ctypes.POINTER(ctypes.c_int)()

        self.lib.ragdoll_graph_detailed_info(ctypes.byref(gid2pid_c), ctypes.byref(num_local_nodes_c), ctypes.byref(gid2lid_unordered_c))

        gid2pid = torch.from_numpy(np.ctypeslib.as_array(gid2pid_c, shape=(n_nodes,))).clone()
        num_local_nodes = torch.from_numpy(np.ctypeslib.as_array(num_local_nodes_c, shape=(world_size,))).clone()
        gid2lid_unordered = torch.from_numpy(np.ctypeslib.as_array(gid2lid_unordered_c, shape=(n_nodes, 2))).clone()
        self.lib.ragdoll_release(gid2pid_c)
        self.lib.ragdoll_release(num_local_nodes_c)
        self.lib.ragdoll_release(gid2lid_unordered_c)

        gid2lid_sorted_key = torch.argsort(gid2lid_unordered[:, 0].view(-1), stable=True)
        gid2lid = torch.gather(gid2lid_unordered[:, 1].view(-1), 0, gid2lid_sorted_key)

        print(gid2pid, num_local_nodes, gid2lid)
        
        return gid2pid, num_local_nodes, gid2lid

    def dispatch_float(self, t, feat_size, local_n_nodes, no_remote=0):
        perf_counter.record_start("dispatch_float_py")
        if t is None:
            t = np.array([])
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        # t = t.ravel().tolist()
        # c_ptr = (ctypes.c_float * len(t))(*t)
        c_ptr = t.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        local_data = (ctypes.c_float * (local_n_nodes * feat_size))()

        perf_counter.record_start("dispatch_float_c")
        self.lib.ragdoll_dispatch_float(
            c_ptr, feat_size, local_n_nodes, local_data, no_remote)
        perf_counter.record_end("dispatch_float_c")
        
        dispatch_res = [local_data[v] for v in range(local_n_nodes * feat_size)]
        perf_counter.record_end("dispatch_float_py")
        return dispatch_res

    def dispatch_int(self, t, feat_size, local_n_nodes, no_remote=0):
        perf_counter.record_start("dispatch_int_py")

        if t is None:
            t = np.array([])
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        # t = t.ravel().tolist()
        # c_ptr = (ctypes.c_int * len(t))(*t)
        c_ptr = t.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        local_data = (ctypes.c_int * (local_n_nodes * feat_size))()
        perf_counter.record_start("dispatch_int_c")
        self.lib.ragdoll_dispatch_float(
            c_ptr, feat_size, local_n_nodes, local_data, no_remote)
        perf_counter.record_end("dispatch_int_c")
        dispatch_res = [local_data[v] for v in range(local_n_nodes * feat_size)]

        perf_counter.record_end("dispatch_int_py")
        return dispatch_res

    def get_local_n_nodes(self):
        return self.lib.ragdoll_get_local_n_nodes()
