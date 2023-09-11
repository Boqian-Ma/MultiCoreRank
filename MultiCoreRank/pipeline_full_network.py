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

data_set = "aarhus"
cap_dataset = "Aarhus"

# data_set = "example"
# cap_dataset = "Example"

# 1 indexed
# layers_to_keep = [3, 10, 11, 12] # 7,9,10,11,12]
#
# layers_to_keep = sorted(layers_to_keep)
# print(layers_to_keep)
# post_fix = "_".join(map(str, layers_to_keep))

print("processing {}".format(data_set))

INPUT_DIR = OUTPUT_DIR = dirname(os.getcwd() + "/../datasets/used_clean_datasets/")
graph = MultilayerGraph(data_set)
print(graph.number_of_nodes)
# find assortavity
print(graph.pearson_correlation_coefficient())

# Run experiments
# 1) assortativity change after removing nodes
# os.system("python assortativity.py {}".format(data_set))

# # 2) inner_most_cores 
os.system("python inner_most_cores.py {}".format(data_set))
# # 3) plot inner_most_cores
# os.system("python inner_most_core_plot.py {} {}".format(data_set + "_0_100_1", cap_dataset))
# # 4) plot number of cores
os.system("python number_of_cores_plot.py {} {}".format(data_set + "_0_100_1", cap_dataset))

# # 5) change in spearman rank
# os.system("python main_rank_spareman.py {}".format(data_set))

# # 6) centrality correlation new (layer aggregate)
# os.system("python centrality_correlation.py {}".format(data_set))

print(graph.get_number_of_edges())