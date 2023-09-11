'''
Iteratively remove nodes and find influence distributions

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

from helpers import create_plot, create_plots, get_influence_node_ranking, get_influence_node_tuples, get_influence_node_tuples_new


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

    # e.g python main.py aps_0_1_6_8_9 o 0.1 5
    
    # data_set = ""
    type = "i"
    # percentage = 0.01
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = 3

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    print_file = PrintFile(data_set)

    # Total removing nodes
    # Find one percent
    one_percent = int(math.floor(0.01 * multilayer_graph.number_of_nodes))

    if one_percent == 0:
        one_percent = 1

    remove_nodes_per_iteration = percentage * one_percent

    print(remove_nodes_per_iteration)
    print("dataset loading time: " + str(time.time()-start_time))

    # Create base plot
    
    # Experiment loop

    if type == "i":
        
        print("First node removal iteration")
        # influences = bfs(multilayer_graph, print_file, False)

        influences = get_influence_node_tuples(multilayer_graph, print_file)
        # full graph influences
        influences_sorted = sorted(influences, key=lambda x: (-x[1], x[0]))        

        curr_nodes = influences

        assert len(multilayer_graph.get_nodes()) == multilayer_graph.modified_number_of_nodes

        # iteration 2

        res = ["removed {}%".format(percentage)]

        # for _ in range(0, 1):
        #     # reset cache
        cache_nodes = curr_nodes
        # remove nodes
        nodes_to_remove = [pair[0] for pair in influences_sorted[:remove_nodes_per_iteration]]

        print(nodes_to_remove)
        print("before removal")
        print(multilayer_graph.modified_number_of_nodes)
        print(len(cache_nodes))

        # TODO Not a bug, but a feature. Need to only consider nodes that are active, remove from list nodes that are in active

        multilayer_graph.remove_nodes(nodes_to_remove)

        print("after removal")
        print(multilayer_graph.modified_number_of_nodes)

        # calculate new ranking 
        influences, _, _ = bfs(multilayer_graph, print_file, False)
        # full graph influences
        influences_sorted = sorted(influences.items(), key=lambda x: (-x[1], x[0]))

        # print(influences)

        curr_nodes = influences.items()

        # new ranking has fewer nodes
        # remove nodes from cache_nodes

        # find intersection of node set, cache and influences, some nodes are removed because no longer active
        # remove nodes that are no longer active in any layers

        # cache has all active nodes in previous layer
        # influences has all nodes active in current layer
        node_difference = find_iso_nodes(cache_nodes, influences.items())
        # remove isolated nodes
        # find union of nodes to remove and isolated nodes
        # remove nodes from cache
        # print(node_difference)
        # print(cache_nodes)

        cache_nodes = remove_items_with_keys(node_difference, cache_nodes)
    
        # print(cache_nodes)
        # print(influences.items())

        assert len(cache_nodes) == len(influences)

        rho = calculate_spareman_correlation(cache_nodes, curr_nodes)

        # print("\n{}\n".format(cache_nodes))

        # print("\n{}\n".format(curr_nodes))

        res.append(rho)

        return res, multilayer_graph.modified_number_of_nodes

def find_iso_nodes(cache_nodes, influences):
    '''
    find union of the first element of two lists of tuples
    '''
    cache_nodes_set = set([pair[0] for pair in cache_nodes])
    curr_nodes_set = set([pair[0] for pair in influences])
    iso_nodes = cache_nodes_set - curr_nodes_set

    return list(iso_nodes)

def calculate_spareman_correlation(cache, curr):

    cache_inf =  [pair[1] for pair in cache]
    curr_inf = [pair[1] for pair in curr]
    rho, _ = stats.spearmanr(cache_inf, curr_inf)

    return rho

def remove_items_with_keys(nodes_to_remove, full_list):
    '''
    Remove a list of tuples from a full list
    '''
    # TODO: optimise
    for node in nodes_to_remove:
        for n in full_list:
            if node == n[0]:
                full_list.remove(n)

    return full_list


if __name__ == "__main__":

    # datasets = ["northamerica_0_2_13_14_11", "southamerica_9_17_21_52"]
    # datasets = ["homo"]
    # datasets = ["moscowathletics2013_multiplex"]
    # datasets = ["celegans"]
    # datasets = ["example"]

    datasets = ["aarhus"]
    # datasets = [sys.argv[1]]

    correlations = []

    for dataset in datasets:
        results = [dataset]
        for i in [1,2,3, 10,20,30]:
            res, num_nodes = main(dataset, i)
            results.append((res,num_nodes) )

        #TODO: clearn up 
        full_path = dirname(os.getcwd()) + "/output/correlation/rank_spareman/{}_30.txt".format(dataset)
        
        if not os.path.exists(os.path.dirname(full_path)):
            try:
                os.makedirs(os.path.dirname(full_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(full_path, 'w+') as f:
            f.write(str(results))

    print(correlations)

    