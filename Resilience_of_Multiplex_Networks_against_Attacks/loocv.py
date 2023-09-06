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


def plot_loocv_area(data, print_file, figure_name, fig_text):

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

    ax.plot(np.concatenate((x_n,x_p)), np.concatenate((negative,positive)), color='C0', linewidth=3)

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

def plot_loocv_line(ours_diff, degree_diff, eigen_diff, betweenness_diff, closeness_diff, print_file, figure_name, fig_text):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = [i for i in range(len(ours_diff))]

    sorted_ours = sorted(ours_diff)
    sorted_degree = sorted(degree_diff)
    sorted_eigen = sorted(eigen_diff)
    sorted_between = sorted(betweenness_diff)
    sorted_close = sorted(closeness_diff)

    combined = np.concatenate((sorted_ours, sorted_between, sorted_degree, sorted_eigen, sorted_close))

    ax.plot(x, sorted_ours, color='C0', linewidth=3,label='Ours', marker='x')
    ax.plot(x, sorted_degree, color='C1', linewidth=3, label='Degree')
    ax.plot(x, sorted_eigen, color='C2', linewidth=3, label='Eigenvector')
    ax.plot(x, sorted_between, color='C3', linewidth=3, label='Betweenness')
    ax.plot(x, sorted_close, color='y', linewidth=3, label='Closeness')

    # center on y=0
    max_abs_y = max(abs(min(combined)), abs(max(combined)))  # maximum absolute y value

    ax.set_ylim([-max_abs_y, max_abs_y])
    ax.axhline(y=0, color='black', linestyle='--')

    ax.legend()
    ax.set_title('Difference between Mean Ranking by LOOCV and One-time Ranking of Various Methods', fontsize='xx-large')
    ax.set_xlabel('Sorted Ranking of Nodes - {}'.format(fig_text), fontsize='x-large')
    ax.set_ylabel('Difference Percentage', fontsize='x-large')
    ax.grid(which='both')

    print_file.print_loocv_cdf_multiple(plt, figure_name)

def main(data_set, print_file):
    '''
    Main function for finding LOOCV of a graph
    '''
    multilayer_graph = MultilayerGraph(data_set)

    our_ranking_dict = initialise_avg_ranking_dict(multilayer_graph.number_of_nodes)
    degree_ranking_dict = initialise_avg_ranking_dict(multilayer_graph.number_of_nodes)
    eigen_ranking_dict = initialise_avg_ranking_dict(multilayer_graph.number_of_nodes)
    betweenness_ranking_dict = initialise_avg_ranking_dict(multilayer_graph.number_of_nodes)
    closeness_ranking_dict = initialise_avg_ranking_dict(multilayer_graph.number_of_nodes)

    # Initial rank of our method
    influence, _, _ = bfs(multilayer_graph, print_file, False)
    initial_rank = sorted(influence.items(), key=lambda x: (-x[1], x[0]))
    update_avg_ranking_dict(initial_rank, our_ranking_dict)
    initial_rank = copy.deepcopy(our_ranking_dict)

    # Initial ranking of degree centrality
    degree_influence = multilayer_graph.overlap_degree_rank()
    degree_initial_rank = sorted(degree_influence.items(), key=lambda x: (-x[1], x[0]))
    update_avg_ranking_dict(degree_initial_rank, degree_ranking_dict)
    degree_initial_rank = copy.deepcopy(degree_ranking_dict)

    # Initial ranking of Eigenvector centrality
    eigen_influence = multilayer_graph.eigenvector_centrality()
    eigen_initial_rank = sorted(eigen_influence.items(), key=lambda x: (-x[1], x[0]))
    update_avg_ranking_dict(eigen_initial_rank, eigen_ranking_dict)
    eigen_initial_rank = copy.deepcopy(eigen_ranking_dict)

    # Initial ranking of betweenness 
    betweenness_influence = multilayer_graph.betweenness_centrality()
    betweenness_initial_rank = sorted(betweenness_influence.items(), key=lambda x: (-x[1], x[0]))
    update_avg_ranking_dict(betweenness_initial_rank, betweenness_ranking_dict)
    betweenness_initial_rank = copy.deepcopy(betweenness_ranking_dict)

    # Initial Closeness ranking 
    closeness_influence = multilayer_graph.closeness_centrality()
    closeness_initial_rank = sorted(closeness_influence.items(), key=lambda x: (-x[1], x[0]))
    update_avg_ranking_dict(closeness_initial_rank, closeness_ranking_dict)
    closeness_initial_rank = copy.deepcopy(closeness_ranking_dict)

    # Calculate LOOCV for each node
    for i in range(multilayer_graph.number_of_nodes):
        # create a deep copy of the graph
        new_graph = copy.deepcopy(multilayer_graph)
        # remove node with index i+1
        new_graph.remove_nodes([i+1])
        
        # Recalculate our method LOOCV ranking
        influence, _, _ = bfs(new_graph, print_file, False)
        sorted_rank = sorted(influence.items(), key=lambda x: (-x[1], x[0]))
        update_avg_ranking_dict(sorted_rank, our_ranking_dict)

        # Recalculate degree method LOOCV ranking
        degree_influence = new_graph.overlap_degree_rank()
        sorted_degree_rank = sorted(degree_influence.items(), key=lambda x: (-x[1], x[0]))
        update_avg_ranking_dict(sorted_degree_rank, degree_ranking_dict)

        # Eigen
        eigen_influence = new_graph.eigenvector_centrality()
        eigen_sorted_rank = sorted(eigen_influence.items(), key=lambda x: (-x[1], x[0]))
        update_avg_ranking_dict(eigen_sorted_rank, eigen_ranking_dict)

        # Betweenness
        betweenness_influence = new_graph.betweenness_centrality()
        betweenness_sorted_rank = sorted(betweenness_influence.items(), key=lambda x: (-x[1], x[0]))
        update_avg_ranking_dict(betweenness_sorted_rank, betweenness_ranking_dict)

        # Closeness
        closeness_influence = new_graph.closeness_centrality()
        closeness_sorted_rank = sorted(closeness_influence.items(), key=lambda x: (-x[1], x[0]))
        update_avg_ranking_dict(closeness_sorted_rank, closeness_ranking_dict)


    # Our method average rank
    avg_ranks = calculate_average_ranking(our_ranking_dict)

    # Degree correlation average rank
    degree_avg_ranks = calculate_average_ranking(degree_ranking_dict)
    eigen_avg_ranks = calculate_average_ranking(eigen_ranking_dict)
    betweenness_avg_ranks = calculate_average_ranking(betweenness_ranking_dict)
    closeness_avg_ranks = calculate_average_ranking(closeness_ranking_dict)


    return (avg_ranks, degree_avg_ranks, eigen_avg_ranks, betweenness_avg_ranks, closeness_avg_ranks), \
        (initial_rank, degree_initial_rank, eigen_initial_rank, betweenness_initial_rank, closeness_initial_rank)
                                           
def calculate_loocv_diff(avg_ranks, initial_rank):
    res = []
    size = len(initial_rank)
    for node in initial_rank.items():
        idx = node[0]
        init_rank = node[1][0]
        avg_rank = avg_ranks[idx]
        diff = (init_rank - avg_rank) / size
        # print(idx, diff*100)
        res.append(diff*100)

    return res

if __name__ == "__main__":
    # datasets = [sys.argv[1]]
    # datasets = ["asia_63", "northamerica_33", "southamerica_13", "europe_75", "celegan"]
    datasets = ["northamerica_33"]
    for dataset in datasets:
        print_file = PrintFile(dataset)
        (avg_ranks, degree_avg_ranks, eigen_avg_ranks, betweenness_avg_ranks, closeness_avg_ranks), \
            (initial_rank, degree_initial_rank, eigen_initial_rank, betweenness_initial_rank, closeness_initial_rank) \
                = main(dataset, print_file)

        '''
        /home/z5260890/Resilience_of_Multiplex_Networks_against_Attacks/venv/bin/python /home/z5260890/Resilience_of_Multiplex_Networks_against_Attacks/Resilience_of_Multiplex_Networks_against_Attacks/loocv.py
        '''

        ours_diff = calculate_loocv_diff(avg_ranks, initial_rank)
        degree_diff = calculate_loocv_diff(degree_avg_ranks, degree_initial_rank)
        eigen_diff = calculate_loocv_diff(eigen_avg_ranks, eigen_initial_rank)

        print(eigen_diff)
        print(eigen_avg_ranks)

        betweenness_diff = calculate_loocv_diff(betweenness_avg_ranks, betweenness_initial_rank)
        closeness_diff = calculate_loocv_diff(closeness_avg_ranks, closeness_initial_rank)

        # plot this shit, check format
        plot_loocv_area(ours_diff, print_file, dataset, "Airlines-NorthAmerica")        

        plot_loocv_line(ours_diff, degree_diff, eigen_diff, betweenness_diff, closeness_diff, print_file, dataset,"Airlines-NorthAmerica")


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

    