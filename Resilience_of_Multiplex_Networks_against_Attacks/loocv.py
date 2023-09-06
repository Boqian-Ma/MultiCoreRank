'''
Leave one out cross validation

Step 1: Calculate ranking of entire network

Step 2: Remove invididual nodes, then calculate ranking of the rest of the nodes.

Step 3: Find average ranking of each node

Step 4: Find Change in ranking avg - normal / total nodes
'''

import argparse
import errno
import math
from posixpath import dirname
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

import numpy as np
from matplotlib import pyplot as plt



# Keeps track of a node's ranking for each iteration
def initialise_avg_ranking_dict(num_nodes):
    ranking = {}
    for i in range(1, num_nodes+1):
        ranking[i] = []
    return ranking

# Add to node ranking dict
def update_avg_ranking_dict(new_ranking, ranking_dict):
    for i in range(len(new_ranking)):
        node = new_ranking[i][0]
        rank = i+1
        ranking_dict[node].append(rank)

# Find average ranking for a node
def calculate_average_ranking(ranking_dict):
    avg_ranks = {}
    for i in ranking_dict.items():

        node = i[0]
        ranks = i[1]
        avg_rank = sum(ranks)/float(len(ranks))

        avg_ranks[node] = avg_rank

    return avg_ranks


# def plot_loocv


def plot_loocv(data, print_file, figure_name):

    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True

    # # sort data

    # x = [i for i in range(len(data))]
    # y = sorted(data)

    # # print(y)
   

    # fig, ax = plt.subplots()

    # # Plot the data
    # ax.plot(x, y, label='wocao', color='blue')

    # # Fill the area underneath the data
    # ax.fill_between(x, y, color='lightblue', alpha=0.4)  # 'alpha' controls the transparency


    # # 1) Draw a vertical line where y = 0
    # # First, find the x value where y = 0 (assuming y is unique for each x)
    # # x_val_at_y0 = x[y.index(0)]
    # # ax.axvline(x_val_at_y0, color='red', linestyle='--', label='y = 0')

    # # 2) Add a grid
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # # Add a legend
    # ax.legend()
    # # Add titles and labels
    # ax.set_title("Plot with filled area underneath")
    # ax.set_xlabel("Nodes")
    # ax.set_ylabel("Percentage of difference in ranking w/o LOOCV")
    fig, ax = plt.subplots(figsize=(12, 6))

    sort = sorted(data)

    positive = [ele for ele in sort if ele >= 0]
    negative = [ele for ele in sort if ele < 0]
    x_p = np.arange(0, len(positive), 1.0)
    x_n = np.arange(-len(negative), 0, 1.0)
    x_axis_sort = np.arange(-len(negative), len(positive), 1.0)

    # Major x ticks every 40, minor x ticks every 10
    # major_sort_ticks = np.concatenate((np.arange(-280, 0, 40), np.arange(0, 181, 40)))
    # minor_sort_ticks = np.concatenate((np.arange(-280, 0, 10), np.arange(0, 181, 10)))

    # ax.plot(x_p, positive, color='C0', linewidth=3)
    # ax.plot(x_n, negative, color='C0', linewidth=3)

    

    ax.plot(x_n + x_p, negative + positive, color='C0', linewidth=3)

    ax.fill_between(x_axis_sort, 0, sort, color='C9')
    ax.axvline(x=0, color='C1')

    # ax.axvline(x=Vline_95_left, color='C3')
    # ax.axvline(x=Vline_95_right, color='C3')
    # ax.axvline(x=-len(negative) - 1, color='C7')
    # ax.axvline(x=len(positive) + 1, color='C7')

    ax.set_title('Difference between Mean Ranking by LOOCV and One-time Ranking', fontsize='xx-large')
    ax.set_xlabel('Sorted Difference Diagram - {}'.format(figure_name), fontsize='x-large')
    ax.set_ylabel('Difference Percentage', fontsize='x-large')

    ax.grid(which='both')
    print_file.print_loocv_cdf(plt, figure_name)


def main(data_set, print_file):
    '''
    Main function for finding LOOCV of a graph
    '''
    # ranking = {}
    # start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)

    ranking_dict = initialise_avg_ranking_dict(multilayer_graph.number_of_nodes)
    print(ranking_dict)

    # Initial rank
    influence, max_level, num_cores = bfs(multilayer_graph, print_file, False)
    initial_rank = sorted(influence.items(), key=lambda x: (-x[1], x[0]))
    update_avg_ranking_dict(initial_rank, ranking_dict)


    initial_rank = copy.deepcopy(ranking_dict)
    # print(initial_rank)
    # print(ranking_dict)

    # Calculate LOOCV for each node
    for i in range(multilayer_graph.number_of_nodes):
        # create a deep copy of the graph
        new_graph = copy.deepcopy(multilayer_graph)
        # remove node with index i+1
        new_graph.remove_nodes([i+1])
        
        influence, _, _ = bfs(new_graph, print_file, False)
        sorted_rank = sorted(influence.items(), key=lambda x: (-x[1], x[0]))

        update_avg_ranking_dict(sorted_rank, ranking_dict)
        # print(influence)


    avg_ranks = calculate_average_ranking(ranking_dict)

    print(avg_ranks)

    return avg_ranks, initial_rank


def calculate_loocv_diff(avg_ranks, initial_rank):

    res = []
    size = len(initial_rank)
    for node in initial_rank.items():
        idx = node[0]
        init_rank = node[1][0]
        avg_rank = avg_ranks[idx]
        diff = (init_rank - avg_rank) / size
        print(idx, diff*100)
        res.append(diff*100)

    return res

if __name__ == "__main__":
    # datasets = [sys.argv[1]]
    datasets = ["aarhus"]
    for dataset in datasets:
        print_file = PrintFile(dataset)
        avg_ranks, initial_ranks = main(dataset, print_file)

        res = calculate_loocv_diff(avg_ranks, initial_ranks)

        

        # plot this shit, check format
        # calculate difference

        # print(avg_ranks)
        # print(initial_rank)
        plot_loocv(res, print_file, dataset)        



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

    