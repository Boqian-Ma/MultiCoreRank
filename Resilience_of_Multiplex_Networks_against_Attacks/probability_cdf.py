import argparse
import math
from resource import error
import time
import copy
from matplotlib import mlab

import numpy as np
import matplotlib
from sklearn.metrics import normalized_mutual_info_score
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from helpers import create_plot, create_plots, get_influence_node_ranking, get_influence, get_influence_node_tuples, get_influence_node_tuples_new

from sklearn.metrics import confusion_matrix


def leave_one_out(data_set, k, print_file):
    '''
    Get current ranking as a list,

    select a top k range

    leave out one node at a time and recalculate influence

    get top k and compare ration of leaving in

    plot n against probability

    '''

    start_time = time.time()
    # Load graph
    multilayer_graph = MultilayerGraph(data_set)
    print("dataset loading time: " + str(time.time()-start_time))


    # Load our full influence to save time
    influences = get_influence_node_tuples_new(multilayer_graph, print_file)
    influence_sorted_by_influence = sorted(influences, key=lambda x: (-x[1], x[0]))
    og_rank_ours = [pair[0] for pair in influence_sorted_by_influence]

    # get og_rank_overlapping_degree
    og_rank_overlapping = multilayer_graph.overlap_degree_rank()
    influence_sorted_by_influence_overlapping = sorted(og_rank_overlapping.items(), key=lambda x: (-x[1], x[0]))
    og_rank_overlap = [pair[0] for pair in influence_sorted_by_influence_overlapping]
    # full_graph_copy = copy.deepcopy(multilayer_graph)

    og_rank_eigen = multilayer_graph.eigenvector_centrality()
    influence_sorted_by_influence_eigen = sorted(og_rank_eigen.items(), key=lambda x: (-x[1], x[0]))
    og_rank_eigen = [pair[0] for pair in influence_sorted_by_influence_eigen]

    recall_list_ours = [0]
    recall_list_overlap = [0]
    recall_list_eigen = [0]

    false_positive_list_ours = [0]
    false_positive_list_overlap = [0]
    false_positive_list_eigen = [0]

    for i in range(10, k):
        # copy graph object 
        # # Find ith node to remove
        # node_to_remove_ours = og_rank_ours[i]
        # print("node to remove")
        # print(node_to_remove_ours)
        # # get rank
        # full_graph_copy.remove_nodes([node_to_remove_ours])

        # # Calculate new influence
        # new_influence = bfs(multilayer_graph, print_file, False)
        # influence_sorted_by_influence = sorted(new_influence.items(), key=lambda x: (-x[1], x[0]))
        # leave_one_out_rank = [pair[0] for pair in influence_sorted_by_influence]

        # # assert len(leave_one_out_rank) == len(og_rank) - 1

        # recall = calculate_recall(og_rank_ours, leave_one_out_rank, i)

        recall_ours, false_positive_ours = ours(multilayer_graph, og_rank_ours, i)
        recall_overlap, false_positive_overlap = overlap(multilayer_graph, og_rank_overlap, i)
        recall_eigen, false_positive_eigen = eigen(multilayer_graph, og_rank_eigen, i)


        recall_list_ours.append(recall_ours)
        recall_list_overlap.append(recall_overlap)
        recall_list_eigen.append(recall_eigen)

        false_positive_list_ours.append(false_positive_ours)
        false_positive_list_overlap.append(false_positive_overlap)
        false_positive_list_eigen.append(false_positive_eigen)

    # assert full_graph_copy.modified_number_of_nodes == (multilayer_graph.number_of_nodes-1)

    # print("og rank")
    # print(og_rank)

    return recall_list_ours, recall_list_overlap, recall_list_eigen, false_positive_list_ours, false_positive_list_overlap, false_positive_list_eigen

def remove_node(full_graph_copy, og_rank, i):
    node_to_remove_ours = og_rank[i]
    # print("node to remove")
    # print(node_to_remove_ours)
    # get rank
    full_graph_copy.remove_nodes([node_to_remove_ours])

def overlap(multilayer_graph, og_rank_overlap, i):
    full_graph_copy = copy.deepcopy(multilayer_graph)
    remove_node(full_graph_copy, og_rank_overlap, i)

    new_rank_overlapping = full_graph_copy.overlap_degree_rank()

    new_influence_sorted_by_influence_overlapping = sorted(new_rank_overlapping.items(), key=lambda x: (-x[1], x[0]))
    leave_one_out_rank_overlapping = [pair[0] for pair in new_influence_sorted_by_influence_overlapping]

    recall, false_positive = calculate_recall(og_rank_overlap, leave_one_out_rank_overlapping, i)

    return recall, false_positive

def eigen(multilayer_graph, og_rank_eigen, i):
    full_graph_copy = copy.deepcopy(multilayer_graph)
    remove_node(full_graph_copy, og_rank_eigen, i)

    new_rank_eigen= full_graph_copy.eigenvector_centrality()

    new_influence_sorted_by_influence_overlapping = sorted(new_rank_eigen.items(), key=lambda x: (-x[1], x[0]))
    leave_one_out_rank_eigen = [pair[0] for pair in new_influence_sorted_by_influence_overlapping]

    recall, false_positive = calculate_recall(og_rank_eigen, leave_one_out_rank_eigen, i)

    return recall, false_positive

def ours(multilayer_graph, og_rank_ours, i):

    # node_to_remove_ours = og_rank_ours[i]
    # print("node to remove")
    # print(node_to_remove_ours)
    # # get rank
    # full_graph_copy.remove_nodes([node_to_remove_ours])
    full_graph_copy = copy.deepcopy(multilayer_graph)

    remove_node(full_graph_copy, og_rank_ours, i)

    # print("get nodes")
    # print(full_graph_copy.get_nodes())
    # Calculate new influence
    new_influence = bfs(full_graph_copy, print_file, False)
    # print("new influence")
    # print(new_influence)
    influence_sorted_by_influence = sorted(new_influence.items(), key=lambda x: (-x[1], x[0]))
    leave_one_out_rank = [pair[0] for pair in influence_sorted_by_influence]
    # assert len(leave_one_out_rank) == len(og_rank) - 1
    recall, false_positive = calculate_recall(og_rank_ours, leave_one_out_rank, i)
    return recall, false_positive

def calculate_recall(old_list, new_list, i):
    # get k elements

    # print(i, old_list, new_list)

    print("\n")

    print(old_list)
    print(new_list)

    print("\n")

    old_list_set = set(old_list[:i])
    new_list_set = set(new_list[:i])

    # print(old_list, new_list)

    # Find side of intersection 
    intersection = old_list_set.intersection(new_list_set)
    recall = len(intersection) / (i + 1.0)

    # print("interseaction")
    # print(intersection)

    # print("k = {} intersection {}".format(i, intersection))

    # print(intersection)

    # find false positive
    difference = old_list_set.difference(new_list_set)
    # print("difference")
    # print(difference)

    # false_positive = len(difference) / (i + 1.0)

    print(old_list[:i], new_list[:i])

    false_positive = false_positive_rate(old_list[:i], new_list[:i])

    return recall, false_positive


def plot_cdf(recall_ours, recall_overlap, recall_eigen, print_file, dataset):
    
    mu = 200
    sigma = 25
    n_bins = 100

    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(recall_ours, n_bins, density=True, histtype='step',
                            cumulative=True, label='Empirical Ours')

    n, bins, patches = ax.hist(recall_overlap, n_bins, density=True, histtype='step',
                            cumulative=True, label='Empirical Overlapping degree')

    n, bins, patches = ax.hist(recall_eigen, n_bins, density=True, histtype='step',
                            cumulative=True, label='Empirical Eigenvector')

    # Add a line showing the expected distribution.
    # y = mlab.normpdf(bins, mu, sigma).cumsum()
    # y /= y[-1]

    # ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

    # Overlay a reversed cumulative histogram.
    # ax.hist(data, bins=bins, density=True, histtype='step', cumulative=-1,
    #         label='Reversed emp.')

    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms {}'.format(dataset))
    ax.set_xlabel('Rank')
    ax.set_ylabel('CDF')

    print_file.print_cdf(plt, dataset)

def calculate_cummulation(recall, false_positive, ax, label):
    # print(normal_array)
    # print("recall")
    # print(recall)
    norm_recall = normalize(recall)
    cdf_recall = np.cumsum(norm_recall)
    # print(cdf)

    # print("false_positive")
    # print(false_positive)
    # norm_false_positive = normalize(false_positive)
    # cdf_false_positive= np.cumsum(norm_false_positive)

    # print(norm_false_positive.sort())
    false_positive = sorted(false_positive)
    x = np.array(false_positive)
    # x = np.array([i + 1 for i in range(len(recall))])
    y = np.array(cdf_recall)
    
    # plt.plot(x, y, label='Empirical Overlapping degree')
    ax.step(x, y, '*', where='post', label=label)

    # ax.hist(norm_recall, 10, density=True, histtype='step',
    #                         cumulative=True, label=label)
    


def normalize(data, target=1.0):
    '''
    Normalise a dictionary
    '''
    norm = []
    raw = sum(data)
    # print(raw)
    factor = target/raw
    # print(factor)
    for node in data:
        norm.append(float(node) * factor)
    return norm

def one_hot(true, pred):

    pred_res = []
    for num in pred:
        if num in true:
            pred_res.append(1)
        else:
            pred_res.append(0)

    true_res = [1 for _ in range(len(true))]

    return true_res, pred_res

def false_positive_rate(true, pred):

    true, pred = one_hot(true, pred)


    print(true, pred)

    print("yeet")

    print(confusion_matrix(true, pred))

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()




    print(fp / (fp + tn))

    return fp / (fp + tn)


if __name__ == "__main__":

    k = 1
    data_set = "aarhus"
    k = 60
    print_file = PrintFile(data_set)
    recall_list_ours, recall_list_overlap, recall_list_eigen, false_positive_list_our, false_positive_list_overlap, false_positive_list_eigen = leave_one_out(data_set, k, print_file)

    # plot_cdf(recall_list_ours, recall_list_overlap, recall_list_eigen, print_file, data_set)

    # print(recall_list_ours, false_positive_list_our)

    fig, ax = plt.subplots(figsize=(8, 4))

    calculate_cummulation(recall_list_ours,false_positive_list_our , ax, 'Empirical Ours')
    calculate_cummulation(recall_list_overlap, false_positive_list_overlap,ax, 'Empirical Overlapping degree')
    calculate_cummulation(recall_list_eigen, false_positive_list_eigen, ax, 'Empirical Eigen Vector')

    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title('Cumulative step histograms {}'.format(data_set))
    ax.set_xlabel('Rank')
    ax.set_ylabel('CDF')

    print_file.print_cdf(plt, data_set)



