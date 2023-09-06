'''
Iteratively remove nodes and find influence distributions

'''

import argparse
import errno
import math
from posixpath import dirname
import random
from resource import error
import sys
import time
import os
from collections import OrderedDict

import copy

import numpy as np
import matplotlib
import pandas as pd

from inner_most_cores.inner_most import inner_most

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from helpers import correlation_mean, create_plot, create_plots, get_influence_node_ranking, get_influence_node_tuples, get_influence_node_tuples_new


from scipy import stats


def main(data_set, percentage, print_file):
    '''
    Main function for finding heatmap and layer-wise pearson correlation
    '''
    # parser = argparse.ArgumentParser(description='Resilience of Multiplex Networks against Attacks')
    # parser.add_argument('d', help='dataset')
    # parser.add_argument('m', help='method: "i"=iterative influence calculation, "o"=once off influence calculation', choices=["i"])
    # parser.add_argument('p', help='percentage of node removal', type=float, choices=np.arange(0.0, 1.0, 0.01))
    # parser.add_argument('c', help='total columns displayed', type=int, choices=range(1, 6))
    # args = parser.parse_args()


    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)
    # Total removing nodes
    # Find one percent
    # new_influence = bfs(multilayer_graph, pint_file, False)

    # inner_most(multilayer_graph, print_file)

    # multilayer_graph.remove_nodes([2])
    # inner_most(multilayer_graph, print_file)
    influence, max_level = bfs(multilayer_graph, print_file, False)

    print(max_level)

def random_attack(multilayer_graph, dataset, removal_percentages, print_file):
    '''
    
    random attack
    - get current set of nodes
    - sample porportion
    - remove nodes
    calculate count

    '''

    # determine count

    # start with index 0
    # num_nodes_to_remove = int(math.floor((removal_percentage/100.0) * multilayer_graph.number_of_nodes))

    # remove 1% at a time
    num_nodes_to_remove = int(math.ceil(multilayer_graph.number_of_nodes/100.0))

    # begin removal process
    map = {}
    # everytime, remove 1% of current nodes
    for removal_percentage in removal_percentages:

        current_nodes = list(multilayer_graph.get_nodes())
        # sample a fixed number of nodes
        try:
            sampled_nodes = random.sample(current_nodes, num_nodes_to_remove)
        except:
            sampled_nodes = current_nodes

        print("\nremoving {} nodes\n".format(num_nodes_to_remove))
        # print("start {} end {}\n".format(cache_num_nodes_to_remove, num_nodes_to_remove))
        print("remaining nodes {}".format(multilayer_graph.modified_number_of_nodes))
        # edge case caused by math.floor
        if num_nodes_to_remove == 0:
            num_nodes_to_remove = 1

        multilayer_graph.remove_nodes(sampled_nodes)

        # find level
        influence , max_level, number_of_cores = bfs(multilayer_graph, print_file, False)

        # influences_sorted = sorted(influence.items(), key=lambda x: (-x[1], x[0]))        
        # og_rank = [pair[0] for pair in influences_sorted]
        # print("max level")
        # print(max_level)

        # add to map
        map[removal_percentage] = (max_level , number_of_cores)

        # stop when graph is empty
        if multilayer_graph.modified_number_of_nodes == 0:
            break

    map = sorted(map.items(), key=lambda x: (x[0]))   
    return map


def new_influence_attack(multilayer_graph, dataset, removal_percentages, print_file):
    # get node ranking initial
    influences = get_influence_node_tuples(multilayer_graph, print_file)
    influences_sorted = sorted(influences, key=lambda x: (-x[1], x[0]))   
    # nodes with highest influence ranked first

    # get a list of influence_node_rank
    influence_node_rank = [node for (node, _) in influences_sorted]

    map = {}
    num_nodes_to_remove = int(math.ceil(multilayer_graph.number_of_nodes/100.0))

    for removal_percentage in removal_percentages:
        # Get current list of nodes
        current_nodes = list(multilayer_graph.get_nodes())

        # update influence_sorted nodes
        for node in influence_node_rank:
            if node not in current_nodes:
                influence_node_rank.remove(node)

        nodes_to_remove = influence_node_rank[:num_nodes_to_remove]
        multilayer_graph.remove_nodes(nodes_to_remove)

        influence , max_level, number_of_cores = bfs(multilayer_graph, print_file, False)

        map[removal_percentage] = (max_level , number_of_cores)

        # stop when graph is empty
        if multilayer_graph.modified_number_of_nodes == 0:
            break

    map = sorted(map.items(), key=lambda x: (x[0]))   
    return map

def func(multilayer_graph, dataset, removal_percentages, print_file):
    '''
    Removal by rank
    '''

    # get full network
    multilayer_graph = MultilayerGraph(dataset)

    # get node ranking initial
    influences = get_influence_node_tuples(multilayer_graph, print_file)
    influences_sorted = sorted(influences, key=lambda x: (-x[1], x[0]))        

    # remove 5% per iteration
    # for iteration in range(1, 21):

    og_rank = [pair[0] for pair in influences_sorted]


    map = {}

    cache_num_nodes_to_remove = int(math.floor((removal_percentages[0]/100.0) * multilayer_graph.number_of_nodes))
    
    for removal_percentage in removal_percentages:
        # print(multilayer_graph.number_of_nodes)
        num_nodes_to_remove = int(math.floor((removal_percentage/100.0) * multilayer_graph.number_of_nodes))
        
        print("\nremoving {} nodes\n".format(num_nodes_to_remove))
        print("start {} end {}\n".format(cache_num_nodes_to_remove, num_nodes_to_remove))
        print("remaining nodes {}".format(multilayer_graph.modified_number_of_nodes))
        
        if num_nodes_to_remove == 0:
            num_nodes_to_remove = 1

        
        # copy_graph = copy.deepcopy(multilayer_graph)
        nodes_to_remove = og_rank[cache_num_nodes_to_remove:num_nodes_to_remove]
        # nodes_to_remove = og_rank[:num_nodes_to_remove]

        # Update cache
        cache_num_nodes_to_remove = num_nodes_to_remove

        # print("nodes to remove")
        # print(nodes_to_remove)
        # quit()

        multilayer_graph.remove_nodes(nodes_to_remove)

        # find level
        influence , max_level, number_of_cores = bfs(multilayer_graph, print_file, False)

        # influences_sorted = sorted(influence.items(), key=lambda x: (-x[1], x[0]))        
        # og_rank = [pair[0] for pair in influences_sorted]

        print("max level")
        print(max_level)

        # add to map
        map[removal_percentage] = (max_level , number_of_cores)

        # stop when graph is empty
        if multilayer_graph.modified_number_of_nodes == 0:
            break

    map = sorted(map.items(), key=lambda x: (x[0]))   

    return map

def build_string(rank):
    s = []
    for x, y in rank:
        s.append("{} {} {}".format(x, y[0], y[1]))
    return "\n".join(s)

if __name__ == "__main__":

    # datasets = ["northamerica_0_2_13_14_11", "southamerica_9_17_21_52"]
    # datasets = ["homo", "biogrid"]
    # datasets = ["moscowathletics2013_multiplex"]
    # datasets = ["northamerica_0_2_13_14_11", "southamerica_9_17_21_52", "celegans", "europe"]

    datasets = ["aarhus"]

    # datasets = [sys.argv[1]]
    start = 0
    finish = 100
    step = 1

    removal_percentages = [i for i in range(start, finish + step, step)]

    print(removal_percentages)

    for dataset in datasets:
        print_file = PrintFile(dataset)

        multilayer_graph = MultilayerGraph(dataset)
        map_influence_attack = new_influence_attack(multilayer_graph, dataset, removal_percentages, print_file)
        influence_attack_output = build_string(map_influence_attack)


        multilayer_graph =  MultilayerGraph(dataset)
        map_random_attack = random_attack(multilayer_graph, dataset, removal_percentages, print_file)
        random_attack_output =  build_string(map_random_attack)

        influence_file_extension = "{}_{}_{}".format(start, finish, step)
        random_file_extension = "{}_{}_{}_random".format(start, finish, step)

        print_file.print_inner_most_core_map_fast(influence_attack_output, influence_file_extension)
        print_file.print_inner_most_core_map_random_attack(random_attack_output, random_file_extension)

        print(map)
        # Save map

    # datasets = ["celegans"]
    # datasets = ["example"]

    # datasets = ["celegans"]
    # correlations = []

    


    # start = 1
    # end = 1

    # runs = [10,20,30]

    # for dataset in datasets:
    #     results = [dataset]
    #     for i in [0]:
    #         res = main(dataset, i)
    #         results.append(res)

    #     #TODO: clearn up 
    #     full_path = dirname(os.getcwd()) + "/output/correlation/assortativity/{}_assortativity_{}.txt".format(dataset, i)
        
    #     if not os.path.exists(os.path.dirname(full_path)):
    #         try:
    #             os.makedirs(os.path.dirname(full_path))
    #         except OSError as exc: # Guard against race condition
    #             if exc.errno != errno.EEXIST:
    #                 raise
    #     with open(full_path, 'w+') as f:
    #         f.write(str(results))

    # print(correlations)

    