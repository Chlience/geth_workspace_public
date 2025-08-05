import os

import dgl
import torch
from loguru import logger
from ogb.nodeproppred import DglNodePropPredDataset


def load_mini_graph_info(
    graph_name, base_dir="/workspace/dataset_process/processed_dataset/"
):
    logger.debug("Loading mini graph info")
    mini_graph_gid2mid = torch.load(
        f"{base_dir}/{graph_name}/partition_result_5000.pt"
    )["partition_assignment"]
    mini_graph = dgl.load_graphs(f"{base_dir}/{graph_name}/coarsened_graph_5000.dgl")[
        0
    ][0]
    mini_graph_info = {
        "n": mini_graph.number_of_nodes(),
        "xadj": mini_graph.adj_external(scipy_fmt="csr").indptr,
        "adjncy": mini_graph.adj_external(scipy_fmt="csr").indices,
        "gid2mid": mini_graph_gid2mid.to(torch.int32).detach().numpy(),
        "node_weights": mini_graph.ndata["size"].to(torch.int32).detach().numpy(),
        "edge_weights": mini_graph.edata["weight"].to(torch.int32).detach().numpy(),
    }
    return mini_graph_info


def get_gnn_dataset(graph_name, base_dir="/workspace/dataset_process/dataset/"):
    """
    Load a graph dataset by name.

    Args:
        graph_name (str): Name of the graph to load
        base_dir (str): Base directory for datasets

    Returns:
        tuple: (graph, num_classes) - DGL graph and number of classes
    """
    if graph_name == "pubmed":
        os.makedirs(f"{base_dir}/pubmed", exist_ok=True)
        dataset = dgl.data.PubmedGraphDataset(raw_dir=f"{base_dir}/pubmed")
        return (
            dataset[0],
            dataset.num_classes,
            dataset[0].ndata["label"].reshape(-1, 1),
            dataset[0].ndata["train_mask"],
            load_mini_graph_info(graph_name),
        )

    elif graph_name == "ogbn-arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv", root=f"{base_dir}")
        graph = dataset[0][0]
        label = dataset[0][1]
        train_idx = dataset.get_idx_split()["train"]
        num_nodes = graph.num_nodes()
        train_mask = torch.zeros([num_nodes], dtype=torch.bool)
        train_mask[train_idx] = True
        return (
            graph,
            dataset.num_classes,
            label,
            train_mask,
            load_mini_graph_info(graph_name),
        )

    elif graph_name == "ogbn-proteins":
        dataset = DglNodePropPredDataset(name="ogbn-proteins", root=f"{base_dir}")
        graph = dataset[0][0]
        label = dataset[0][1]
        train_idx = dataset.get_idx_split()["train"]
        num_nodes = graph.num_nodes()
        train_mask = torch.zeros([num_nodes], dtype=torch.bool)
        train_mask[train_idx] = True
        return (
            graph,
            dataset.num_classes,
            label,
            train_mask,
            load_mini_graph_info(graph_name),
        )

    elif graph_name == "yelp":
        os.makedirs(f"{base_dir}/yelp", exist_ok=True)
        dataset = dgl.data.YelpDataset(raw_dir=f"{base_dir}/yelp")
        return (
            dataset[0],
            dataset.num_classes,
            dataset[0].ndata["label"],
            dataset[0].ndata["train_mask"],
            load_mini_graph_info(graph_name),
        )

    elif graph_name == "reddit":
        dataset = dgl.data.RedditDataset(raw_dir=f"{base_dir}/reddit")
        dataset.load()
        return (
            dataset[0],
            dataset.num_classes,
            dataset[0].ndata["label"].reshape(-1, 1),
            dataset[0].ndata["train_mask"],
            load_mini_graph_info(graph_name),
        )

    elif graph_name == "ogbn-products":
        dataset = DglNodePropPredDataset(name="ogbn-products", root=f"{base_dir}")
        graph = dataset[0][0]
        label = dataset[0][1]
        train_idx = dataset.get_idx_split()["train"]
        num_nodes = graph.num_nodes()
        train_mask = torch.zeros([num_nodes], dtype=torch.bool)
        train_mask[train_idx] = True
        return (
            graph,
            dataset.num_classes,
            label,
            train_mask,
            load_mini_graph_info(graph_name),
        )

    elif graph_name == "ogbn-papers100M":
        dataset = DglNodePropPredDataset(name="ogbn-papers100M", root=f"{base_dir}")
        graph = dataset[0][0]
        label = dataset[0][1]
        train_idx = dataset.get_idx_split()["train"]
        num_nodes = graph.num_nodes()
        train_mask = torch.zeros([num_nodes], dtype=torch.bool)
        train_mask[train_idx] = True
        return (
            graph,
            dataset.num_classes,
            label,
            train_mask,
            load_mini_graph_info(graph_name),
        )

    else:
        raise ValueError(f"Unknown graph dataset: {graph_name}")
