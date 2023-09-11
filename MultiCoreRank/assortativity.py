'''
Find network assortativity after influential nodes are removed
'''
import argparse
import errno
import math
from posixpath import dirname
from resource import error
import sys
import time
import os
import copy
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from helpers import correlation_mean, create_plot, create_plots, get_influence_node_ranking, get_influence_node_tuples
from scipy import stats


def main(data_set, percentage):
    '''
    Main function for finding heatmap and layer-wise pearson correlation
    '''
    # parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks')
    # parser.add_argument('d', help='dataset')
    # parser.add_argument('m', help='method: "i"=iterative influence calculation, "o"=once off influence calculation', choices=["i"])
    # parser.add_argument('p', help='percentage of node removal', type=float, choices=np.arange(0.0, 1.0, 0.01))
    # parser.add_argument('c', help='total columns displayed', type=int, choices=range(1, 6))
    # args = parser.parse_args()

    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    print_file = PrintFile(data_set)

    # Total removing nodes
    # Find one percent
    one_percent = int(math.floor(0.01 * multilayer_graph.number_of_nodes))

    if one_percent == 0:
        one_percent = 1

    if percentage == 0:
        one_percent = 0

    remove_nodes_per_iteration = percentage * one_percent


    print("First node removal iteration")
    # influences = bfs(multilayer_graph, print_file, False)

    influences = get_influence_node_tuples(multilayer_graph, print_file)
    # full graph influences

    influences_sorted = sorted(influences, key=lambda x: (-x[1], x[0]))        


    assert len(multilayer_graph.get_nodes()) == multilayer_graph.modified_number_of_nodes

    # iteration 2

    nodes_to_remove = [pair[0] for pair in influences_sorted[:remove_nodes_per_iteration]]

    multilayer_graph.remove_nodes(nodes_to_remove)

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()

    # flatten list for density plots, not including nan values
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist if not math.isnan(item)]

    # Calculate mean
    mean_diag, mean_no_diag = correlation_mean(pearson_flat_list, multilayer_graph.number_of_layers)
    
    res = ["removed {}%: {}, remaining nodes: {}".format(percentage, mean_no_diag, multilayer_graph.modified_number_of_nodes)]

    return res


if __name__ == "__main__":

    # datasets = ["northamerica_0_2_13_14_11", "southamerica_9_17_21_52"]
    # datasets = ["homo", "biogrid"]
    # datasets = ["moscowathletics2013_multiplex"]
    # datasets = ["northamerica_0_2_13_14_11", "southamerica_9_17_21_52", "celegans", "europe", "homo", "biogrid"]
    # datasets = ["celegans"]
    # datasets = ["example"]

    # datasets = ["southamerica_0_2_3_4_5_6_7_8_9_10_11_12_13_14_16_17_20_21_22"]

    # datasets = ["southamerica_0_2_4_5_6_8_9_10_12_13_14_16_17_20_21"]

    # datasets = ["southamerica_0_2_4_5_6_8_9_10_12_14_16_17_20"]

    # datasets = ["southamerica_0_2_4_5_6_8_9"]

    # datasets = ["arxiv_netscience_multiplex_3_11"]

    # datasets = ["arxiv_netscience_multiplex"]

    # datasets = [sys.argv[1]]
    datasets = ["celegans"]
    correlations = []
    # start = 1
    # end = 1

    # Removal percentage
    runs = [0,1,2,3,10,20,30]

    for dataset in datasets:
        results = [dataset]
        for i in runs:
            res = main(dataset, i)
            results.append(res)

        #TODO: clearn up 
        full_path = dirname(os.getcwd()) + "/output/correlation/assortativity/{}_assortativity_{}.txt".format(dataset, i)
        
        if not os.path.exists(os.path.dirname(full_path)):
            try:
                os.makedirs(os.path.dirname(full_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(full_path, 'w+') as f:
            f.write(str(results))

    print(results[1])
    # print(correlations)

    