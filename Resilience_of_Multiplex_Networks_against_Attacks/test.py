'''
Iteratively remove nodes and find influence distributions

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
    start_time = time.time()
    # Load graph
    print(data_set)
    multilayer_graph = MultilayerGraph(data_set)
    # Total removing nodes
    # Find one percent
    # new_influence = bfs(multilayer_graph, pint_file, False)

    # inner_most(multilayer_graph, print_file)

    # multilayer_graph.remove_nodes([2])
    # inner_most(multilayer_graph, print_file)
    influence, max_level = bfs(multilayer_graph, print_file, False)

    print(len(influence))

    print(time.time() - start_time)

if __name__ == "__main__":

    # datasets = ["celegans"]

    # start = 0
    # finish = 20
    # step = 2

    # qsub -I -l select=1:ncpus=40:mem=100gb,walltime=5:00:00
    # removal_percentages = [i for i in range(start, finish + step, step)]

    # print(removal_percentages)

    # for dataset in datasets:
    #     print_file = PrintFile(dataset)
    #     multilayer_graph = MultilayerGraph(dataset)

    #     map = func(multilayer_graph, dataset, removal_percentages, print_file)
    #     s = build_string(map)

    #     file_extension = "{}_{}".format(start, finish)

    #     print_file.print_inner_most_core_map(s, file_extension)

    dataset = "homo"
    print_file = PrintFile(dataset)
    main(dataset, 1, print_file)
    


    