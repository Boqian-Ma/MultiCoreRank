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

data_set = "europe"
total_layers = 175

# ranks = get_pos_layers(data_set)

ranks = get_high_inf_layers(data_set)

# length = len(ranks)

# print(length)


ranks = [i+1 for i in ranks]

# print(ranks)

layers_to_keep = [i for i in range(1, total_layers+1)]

# layers_to_keep = [1,3]

# North america
# layers_to_remove = [9,17,24,49,28,51,61,53,58,75, 76, 35, 39, 34, 44, 99, 16, 25, 93, 7, 80, 21, 82, 55, 59, 6, 57, 68, 8, 54, 62,
# 84, 4, 2, 38, 118, 18, 63, 83, 5, 11, 19, 56, 72]

# layers_to_remove = list(reversed(ranks))[:5]

# layers_to_remove = list(reversed(ranks))[:10]
layers_to_remove = ranks[:100]

print(layers_to_remove)

# quit()

# asia
# layers_to_remove = [46, 2, 23, 57, 24, 37, 61, 28, 5, 31, 29, 44, 53, 8, 15, 17, 20, 40, 36, 12, 49, 59, 25]


for l in layers_to_remove:
    if l in layers_to_keep:
        layers_to_keep.remove(l)

layers_to_keep = sorted(layers_to_keep)


# print(layers_to_keep)

# post_fix = "_".join(map(str, layers_to_keep))


post_fix = str(len(layers_to_keep))

# data_set = "northamerica"
cap_dataset = "Europe" #+ str(post_fix)

print("processing {}".format(data_set + "_" + post_fix))
print( data_set + "_" + post_fix)

INPUT_DIR = OUTPUT_DIR = dirname(os.getcwd() + "/../datasets/used_clean_datasets/")

full_dataset = ""
input_path = ""

# select layers
# layers_to_keep = []

# open file

node_map = {}
new_lines = []
dataset_file = open(os.path.join(INPUT_DIR, data_set + ".txt"))

# Network data
first_line = dataset_file.readline()
split_first_line = first_line.split(' ')
num_layers, num_nodes = split_first_line[0], split_first_line[1]

print(num_layers, num_nodes)


print("remap....")

# create string with layers to keep
# Process lines
i = 0

new_lines = []
node_map = {}

layer_map = {}

for line in dataset_file:
    layer, source, dest = line.strip("\n").split(" ")
    # source, dest = int(source), int(dest)
    
    # print(layer, source, dest)
    # print(int(layer))

    if int(layer) in layers_to_keep:
        # print("fuck")
        if source not in node_map:
            # starting index = 1
            node_map[source] = str(len(node_map) + 1) # add a new map
        if dest not in node_map:
            # starting index = 1
            node_map[dest] = str(len(node_map) + 1)
        # Create new file

        if int(layer) not in layer_map.keys():
            layer_map[int(layer)] = len(layer_map.keys()) + 1

        new_lines.append("{} {} {}\n".format(layer, node_map[source], node_map[dest]))

dataset_file.close()
# output file

print(len(new_lines))

num_layers = len(layers_to_keep) 

with open(os.path.join(OUTPUT_DIR, data_set + "_" + post_fix + ".txt"), "w+") as output_file:
        output_file.write("{} {} {}\n".format(num_layers, len(set(node_map.keys())), len(set(node_map.keys()))))
        output_file.writelines(new_lines)

# load data from output file

print("new file: num nodes {}".format(len(set(node_map.keys()))))


# os.system("python temp.py {}".format(data_set + "_" + post_fix))

dataset = data_set + "_" + post_fix

graph = MultilayerGraph(dataset)

# print(graph.number_of_nodes)

# # find assortavity
# print(graph.pearson_correlation_coefficient())

# Run experiments
# 1) assortativity change after removing nodes
# os.system("python assortativity.py {}".format(dataset))

# 2) inner_most_cores 
# os.system("python inner_most_cores.py {}".format(dataset))

# # # 3) plot inner_most_cores
os.system("python inner_most_core_plot.py {} {}".format(dataset + "_0_100_1", cap_dataset))

# # # 4) plot number of cores
os.system("python number_of_cores_plot.py {} {}".format(dataset + "_0_100_1", cap_dataset))

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