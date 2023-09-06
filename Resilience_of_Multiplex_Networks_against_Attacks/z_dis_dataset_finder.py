import argparse
import itertools
import json
import math
import operator
from scipy.stats import spearmanr
from multilayer_graph.multilayer_graph import MultilayerGraph
from utilities.print_file import PrintFile 
from helpers import get_influence_node_tuples
from helpers import correlation_mean, create_plot, create_plots, get_influence_node_ranking, get_influence_node_tuples

import random

import time

import numpy as np

import os
import sys

import collections

def create_new_txt_data(data_set, layers_to_keep, INPUT_DIR):
    '''
    create and remap new dataset with layers required
    '''
    OUTPUT_DIR = INPUT_DIR

    layers_to_keep = sorted(layers_to_keep)
    print(layers_to_keep)
    post_fix = "_".join(map(str, layers_to_keep))
    print("processing {}".format(data_set + "_" + post_fix))

    # open file

    node_map = {}
    new_lines = []
    dataset_file = open(os.path.join(INPUT_DIR, data_set + ".txt"))

    # Network data
    first_line = dataset_file.readline()
    split_first_line = first_line.split(' ')
    num_layers, num_nodes = split_first_line[0], split_first_line[1]

    # print(num_layers, num_nodes)

    print("remaping....")

    # create string with layers to keep
    # Process lines

    new_lines = []
    node_map = {}

    layers = []

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

            # 1 index
            if int(layer) not in layer_map.keys():
                layer_map[int(layer)] = len(layer_map.keys()) + 1

            new_lines.append("{} {} {}\n".format(layer_map[int(layer)], node_map[source], node_map[dest]))

    dataset_file.close()
    # output file

    # print(len(new_lines))

    num_layers = len(layers_to_keep) 

    with open(os.path.join(OUTPUT_DIR, data_set + "_" + post_fix + ".txt"), "w+") as output_file:
            output_file.write("{} {} {}\n".format(num_layers, len(set(node_map.keys())), len(set(node_map.keys()))))
            output_file.writelines(new_lines)

    return data_set + "_" + post_fix

def main(data_set):

    INPUT_DIR = OUTPUT_DIR = os.path.dirname(os.getcwd() + "/../datasets/disassortative_datasets/garbage_northamerica/")


    res = {} # layer tuple: assortavity

    # layers_to_keep = [layer_1, layer_2] 

    multilayer_graph = MultilayerGraph(data_set)
    # dis_layers, count = multilayer_graph.pearson_correlation_coefficient_find_negatives()

    # count = count.items()
    # # sort number by number of disassortative layers
    # count.sort(key=lambda x: -x[1])

    # # print(count)

    # layers = [pair[0] for pair in count]

    layers = [i for i in range(multilayer_graph.number_of_layers - 120)]


    print(len(layers))
    print("yeet")

    # layers had all layers

    combinations = list(itertools.combinations(layers, 15))

    random.shuffle(combinations)

    # with open(os.path.join(OUTPUT_DIR, "a_"+ file_name + ".txt"), "w+") as f:
    #     res = res.items()
    #     res.sort(key=lambda x: -x[1])
    #     res = [str(i) + "\n" for i in res]
    #     f.writelines(res)

    # print("combo")

    # file = open(os.path.join(OUTPUT_DIR, "a_"+ "northamerica" + ".txt"), "w+")


    for i in range(len(combinations)):
        layers_to_keep = combinations[i]

        # print(layers_to_keep)
        # create dataset 
        file_name = create_new_txt_data(data_set, layers_to_keep, INPUT_DIR)
        # load graph and find assortativity

        graph = MultilayerGraph(file_name)

        pearson_coe_matrix = graph.pearson_correlation_coefficient()

        print(pearson_coe_matrix)

        pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist if not math.isnan(item)]

        print(graph.number_of_layers)
        print(pearson_flat_list)

        # quit()
        # Calculate mean

        check = check_ratio(pearson_flat_list, graph.number_of_layers)

        # print(check)
        mean_diag, mean_no_diag = correlation_mean(pearson_flat_list, graph.number_of_layers)
    
        # print(mean_no_diag)    
        if mean_no_diag < 0 and check:
            res[str(tuple(layers_to_keep))] = mean_no_diag

            # layer_string = "_".join(map(str,layers_to_keep))
            # s = "{} {}\n".format(layer_string, mean_no_diag)
            # file.write(s)

    with open(os.path.join(OUTPUT_DIR, "a_"+ "northamerica" + ".txt"), "w+") as f:
        res = res.items()
        res.sort(key=lambda x: -x[1])
        res = [str(i) + "\n" for i in res]
        f.writelines(res)

def check_ratio(flat_list, num_layers):
    length = float(len(flat_list) - num_layers)
    count = 0.0
    for i in flat_list:
        if i == 1:
            continue
        if i < 0:
            count += 1

    # print("ratio {}".format(count/length))
    if count/length > 0.8:
        return True
    

    return False

if __name__ == "__main__":
    # main()
    # # datasets = ["biogrid", "celegans", "example", "homo", "oceania", "sacchcere", "aps", "northamerica"]
    # datasets = ["europe", "dblp", "asia", "amazon", "biogrid", "celegans", "example", "homo", "oceania", "sacchcere", "aps", "northamerica"]
    # datasets = ["europe", "asia", "oceania", "northamerica", "southamerica"]
    # print("hell yearh")
    
    datasets = ["europe"]
    # for data_set in datasets:
    #     sum_rows(data_set)

    for data_set in datasets:
        main(data_set)
    
        