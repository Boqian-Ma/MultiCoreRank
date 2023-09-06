import argparse
import operator
from scipy.stats import spearmanr
from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from helpers import get_influence_node_tuples

import time

import numpy as np

import collections


def main(data_set):

    print("Finding disassortative layers")
    # parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks: centrality methods correlation')
    # parser.add_argument('d', help='dataset')
    # args = parser.parse_args()
    # data_set = "southamerica"

    print_file = PrintFile(data_set)

    # Load graph
    multilayer_graph = MultilayerGraph(data_set)
    # dis_layers, count = multilayer_graph.pearson_correlation_coefficient_find_negatives()

    dis_layers, count = multilayer_graph.pearson_correlation_coefficient_find_positives()

    count = count.items()
    # sort number by number of disassortative layers
    count.sort(key=lambda x: -x[1])

    # print(corr_matrix)
    # print(np.asarray(corr_matrix).T)

    # print_file.print_negative_correlation_layers(dis_layers)

    print_file.print_positive_correlation_layers(dis_layers)

    res = []

    for i in count:
        res.append("{}\n".format(str(i)))
    
    # print_file.print_count_dis_layers(res)

    print_file.print_count_pos_layers(res)

def sum_rows(data_set):
    print("Finding disassortative layers")
    # parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks: centrality methods correlation')
    # parser.add_argument('d', help='dataset')
    # args = parser.parse_args()
    # data_set = "aps"

    print_file = PrintFile(data_set)
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)
    corr_matrix = multilayer_graph.pearson_correlation_coefficient()
    sums = {}

    for i in range(len(corr_matrix)):
        # +1 for redability
        sums[i] = (sum(corr_matrix[i]) - 1) / (len(corr_matrix[i]) - 1) # -1 because we ignore the correlation of a layer with itself

    sorted_x = sorted(sums.items(), key=operator.itemgetter(1))
    sorted_dict = collections.OrderedDict(sorted_x)

    print(sorted_x)
    print_file.print_average_correlation_layers(sorted_dict)

if __name__ == "__main__":
    # main()
    # # datasets = ["biogrid", "celegans", "example", "homo", "oceania", "sacchcere", "aps", "northamerica"]
    # datasets = ["europe", "dblp", "asia", "amazon", "biogrid", "celegans", "example", "homo", "oceania", "sacchcere", "aps", "northamerica"]
    # datasets = ["europe", "asia", "oceania", "northamerica", "southamerica"]
    # datasets = ["asia", "europe", "southamerica"]
    datasets = ["arxiv_netscience_multiplex"]
    # for data_set in datasets:
    #     sum_rows(data_set)
    for dataset in datasets:
        main(dataset)
