import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from core_decomposition.breadth_first_v3 import breadth_first as bfs

def correlation_mean(list, num_layers):
    '''
    Calculate mean of a list
    '''
    mean_diag = sum(list) / float(len(list))

    try:
        mean_no_diag = (sum(list) -  num_layers) / float(len(list) -  num_layers)
    
    except:
        mean_no_diag = float("NAN")
    
    return mean_diag, mean_no_diag

def create_plots(multilayer_graph, plot_col, axs):
    '''
    Create density distribution plot and heatmap
    Add to the existing set of plots
    '''
    # Calculate pearson coefficients
    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()

    # flatten list for density plots, not including nan values
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist if not math.isnan(item)]

    # heat map
    im = axs[0, plot_col].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)
    plt.colorbar(im, ax=axs[0, plot_col])

    # Calculate mean
    mean_diag, mean_no_diag = correlation_mean(pearson_flat_list, multilayer_graph.number_of_layers)
    
    # Print mean values
    at = AnchoredText("Mean including diag: {:.2f}\nMean excluding diag: {:.2f}".format(mean_diag, mean_no_diag), prop=dict(size=15), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[0, plot_col].add_artist(at)

    # normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    hist = axs[1, plot_col].hist(pearson_flat_list, bins=10, weights=weights)
    axs[1, plot_col].set_xlim(-1, 1.5)
    axs[1, plot_col].set_ylabel('Density')
    axs[1, plot_col].set_xlabel('Pearson Correlation Coefficient')
    axs[1, plot_col].axvline(mean_no_diag, color='k', linestyle='dashed', linewidth=3)

    return None

def create_plot(multilayer_graph, axs):

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]
    
    # heat map
    im = axs[0].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)    

    plt.colorbar(im, ax=axs[0])

    mean_diag, mean_no_diag = correlation_mean(pearson_flat_list, len(pearson_coe_matrix))

    # Print mean values
    at = AnchoredText("Mean including diag: {:.2f}\nMean excluding diag: {:.2f}".format(mean_diag, mean_no_diag), prop=dict(size=15), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[0].add_artist(at)

    # Normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    hist = axs[1].hist(pearson_flat_list, bins=10, weights=weights)

    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel('Density')
    axs[1].set_xlabel('Pearson Correlation Coefficient')
    axs[1].axvline(mean_no_diag, color='k', linestyle='dashed', linewidth=3)


def tuple_list_to_dict(tup):

    pass

def get_influence_node_ranking(multilayer_graph, print_file):
    '''
    Get node ranking from file
    '''
    if not os.path.isfile(print_file.full_influence_rank_file):
        # calculate influence and save output
        print("")
        influence = bfs(multilayer_graph, print_file, False)
        # Unpack tuple
        # where to sort?
        influence_sorted_by_influence = sorted(influence.items(), key=lambda x: (-x[1], x[0]))
        rank = [pair[0] for pair in influence_sorted_by_influence]
    else:
        #load influence rank by reading file
        rank = []
        with open(print_file.full_influence_rank_file, 'r') as f:
            for line in f:
                rank.append(line.strip().split("\t")[0])

        if len(rank) != multilayer_graph.number_of_nodes:
            raise ValueError("influence ranking file is incomplete: length of given file is different from length of graph nodes")
        rank = list(map(int, rank))    
    return rank


def get_influence_node_tuples(multilayer_graph, print_file):
    '''
    Get node ranking from file
    returns a dict sorted by key
    '''
    if not os.path.isfile(print_file.full_influence_rank_file):
        # calculate influence and save output
        # print("")
        influence, _, _ = bfs(multilayer_graph, print_file, False)
        influence = [(k, v) for k, v in influence.items()]

    else:
        #load influence rank by reading file
        influence = []
        with open(print_file.full_influence_rank_file, 'r') as f:
            for line in f:
                node_inf = line.strip().split("\t")
                influence.append((int(node_inf[0]), float(node_inf[1])))        
        
        # print(influence)
        
        if len(influence) != multilayer_graph.number_of_nodes:
            raise ValueError("influence ranking file is incomplete: length of given file is different from length of graph nodes")
        influence.sort(reverse=False)

    return influence

# TODO fix
def get_influence_node_tuples_new(multilayer_graph, print_file):
    '''
    Get node ranking from file
    returns a dict sorted by key
    '''
    if not os.path.isfile(print_file.full_influence_rank_file_new):
        # calculate influence and save output
        # print("")
        influence = bfs(multilayer_graph, print_file, False)
        influence = [(k, v) for k, v in influence.items()]
        # print("\nfuck me 2\n")

        # need to save
    else:
        #load influence rank by reading file
        influence = []
        # print("\nfuck me 1\n")
        with open(print_file.full_influence_rank_file_new, 'r') as f:
            for line in f:
                node_inf = line.strip().split("\t")
                influence.append((int(node_inf[0]), float(node_inf[1])))        
        if len(influence) != multilayer_graph.number_of_nodes:
            raise ValueError("influence ranking file is incomplete: length of given file is different from length of graph nodes")
        influence.sort(reverse=False)
    
    return influence    

def get_influence(multilayer_graph, print_file):
    '''
    Get a list of node influence from file. Only influence. 
    '''
    if not os.path.isfile(print_file.full_influence_rank_file):
        # calculate influence and save output
        # print("")
        influence = bfs(multilayer_graph, print_file, False).values()
    else:
        #load influence rank by reading file
        influence = []
        with open(print_file.full_influence_rank_file, 'r') as f:
            for line in f:
                node_inf = line.strip().split("\t")
                influence.append(float(node_inf[1])) 
        if len(influence) != multilayer_graph.number_of_nodes:
            raise ValueError("influence ranking file is incomplete: length of given file is different from length of graph nodes")
    return influence