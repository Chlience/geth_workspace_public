from geth.data.graph_dataset import get_gnn_dataset

if __name__ == "__main__":
    datasets = ["ogbn-arxiv", "ogbn-products", "reddit", "yelp"]
    for dataset in datasets:
        print(f"Download dataset: {dataset}")
        dataset = get_gnn_dataset(dataset)
    