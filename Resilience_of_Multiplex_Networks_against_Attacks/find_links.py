import argparse
from audioop import mul
# from scipy.stats import spearmanr
from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from helpers import get_influence_node_tuples


def main(dataset):
    # load graph
    # get num link

    multilayer_graph = MultilayerGraph(dataset)
    num_edges = multilayer_graph.get_number_of_edges()
    num_nodes = multilayer_graph.number_of_nodes
    # print("Dataset name {}, #links {}".format(dataset, num_edges))
    return (dataset, num_edges, num_nodes)


if __name__ == "__main__":
    # datasets = ["celegans", "homo", "biogrid", "europe"]

    datasets = ["northamerica_0_2_13_14_11", "southamerica_9_17_21_52"]

    # datasets/used_clean_datasets/southamerica_9_17_21_51.txt

    res = []
    for dataset in datasets:
        res.append(main(dataset))

    for i in res:
        print(i)