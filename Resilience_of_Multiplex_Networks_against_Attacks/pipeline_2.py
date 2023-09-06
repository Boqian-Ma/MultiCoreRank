'''
Non full graph
'''


import argparse
import errno
import math
from posixpath import dirname
from resource import error
import time
import os
from collections import OrderedDict
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

# layer_map [0, 4, 9, 10, 11]

# data_set = "fao_trade_multiplex"
# cap_dataset = "FAO Trade Network"

# 1 indexed
# layers_to_keep = [3, 10, 11, 12] # 7,9,10,11,12] arxiv

# layers_to_keep = [1, 5, 11]
# layers_to_keep = [1, 8, 9]
# 0 1 2 3 4 5 6 7 9 13

# layers_to_keep = [1,2]

# layers_to_keep = [1,2,3,4,5,6,7,8,10,14]

def get_pos_layers(dataset):

    path = "/home/z5260890/Resilience_of_Multiplex_Networks_against_Attacks/output/disassortative_layers/{}_pos_layers_count.txt".format(dataset)

    ranks = []
    with open(path, "r") as f:
        data = f.read()

    items = data.split("\n")

    for item in items:
        rank = item[1:-1].split(" ")

        if len(rank) == 2:
            ranks.append(rank[0][:-1])

    return map(int, ranks)

def get_high_inf_layers(dataset):

    path = "/home/z5260890/Resilience_of_Multiplex_Networks_against_Attacks/output/disassortative_layers/{}_pos_layers.txt".format(dataset)

    ranks = []
    with open(path, "r") as f:
        data = f.read()

    items = data.split("\n")
    
    for item in items:
        rank = item.split(" ")

        if len(rank) == 3:
            ranks.append(rank[0])

    l =  map(int, ranks)

    return list(OrderedDict.fromkeys(l))

# arxiv, asia

data_set = "aarhus"
total_layers = 13

# ranks = get_pos_layers(data_set)

cap_dataset = "Aarhus" #+ str(post_fix)

graph = MultilayerGraph(data_set)

# print(graph.number_of_nodes)

# # find assortavity
# print(graph.pearson_correlation_coefficient())

# Run experiments
# 1) assortativity change after removing nodes
# os.system("python assortativity.py {}".format(dataset))

# 2) inner_most_cores 
# os.system("python inner_most_cores.py {}".format(dataset))

# # # 3) plot inner_most_cores
# os.system("python inner_most_core_plot.py {} {}".format(dataset + "_0_100_1", cap_dataset))

# # # 4) plot number of cores
os.system("python number_of_cores_plot.py {} {}".format(data_set + "_0_100_1", cap_dataset))

# # # # # 5) change in spearman rank
# os.system("python main_rank_spareman.py {}".format(dataset))

# # # # # 6) centrality correlation new (layer aggregate)
# os.system("python centrality_correlation.py {}".format(dataset))

# # 7) assortativity
# os.system("python assortativity.py {}".format(dataset))

print("number of nodes")
print(graph.number_of_nodes)
print("number of edges")
print(graph.get_number_of_edges())

print(dataset)
print(dataset + "_0_100_1",cap_dataset)