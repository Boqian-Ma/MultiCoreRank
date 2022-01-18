import argparse
import math
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

def create_plots(multilayer_graph, plot_col, axs, density_y_lim=None):
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
    mean_diag = sum(pearson_flat_list) / float(len(pearson_flat_list))

    try:
        mean_no_diag = (sum(pearson_flat_list) -  multilayer_graph.number_of_layers) / float(len(pearson_flat_list) -  multilayer_graph.number_of_layers)
    except:
        mean_no_diag = float("NAN")
    
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

    axs[1, plot_col].axvline(mean_no_diag, color='k', linestyle='dashed', linewidth=1)

    return None

def create_plot(multilayer_graph, axs):

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]
    
    # heat map
    im = axs[0].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)    

    plt.colorbar(im, ax=axs[0])

    # Normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    hist = axs[1].hist(pearson_flat_list, bins=10, weights=weights)

    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel('Density')
    axs[1].set_xlabel('Pearson Correlation Coefficient')

    # sns.kdeplot(bin_centers, ax=axs[2])
    # Check area underneith
    # widths = bins[1:] - bins[:-1]
    #assert((density * widths).sum() == 1.0)
    # print((density * widths).sum())

def main():
    start_time = time.time()
    data_set = "sacchcere"
    percentage = 0.2
    # number of columns in the final output
    # total_columns - 1 is the number of times the percentage
    total_columns = 5

    multilayer_graph = MultilayerGraph(data_set)

    print("loading time: " + str(time.time()-start_time))

    # find out how many graphs
    fig, axs = plt.subplots(2, total_columns, figsize=(40, 20))

    # Plotting multiple
    if total_columns > 1:
        # first column

        axs[0, 0].set_title("Full network")
        err = create_plots(multilayer_graph, 0, axs)

        # First plot
        density_y_lim = axs[1, 0].get_ylim()
        
        # -1 because the first column if the entire graph, then the next 5 are results of node removal
        total_num_remove_nodes = math.floor(percentage * multilayer_graph.number_of_nodes)
        remove_nodes_per_iteration = int(math.ceil(total_num_remove_nodes / (total_columns - 1)))

        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \nTotal Node removal percentage: {}%\nTotal removing nodes: {}, Per iteration # of node removal: {}'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers ,percentage * 100 ,total_num_remove_nodes, remove_nodes_per_iteration), fontsize=16)
        print("First iteration")
        
        for col in range(1, total_columns):
            # find influence
            influence = bfs(multilayer_graph, PrintFile(data_set), False, data_set)

            print("iteration {} done....".format(col))

            nodes_to_remove = [pair[0] for pair in influence[:remove_nodes_per_iteration]]
            # remove nodes
            multilayer_graph.remove_nodes(nodes_to_remove)
            # find new plots
            err = create_plots(multilayer_graph, col, axs, density_y_lim=density_y_lim)

            if err:
                break

            axs[0, col].set_title("Iteration {}, Remaining nodes: {}".format(col, multilayer_graph.modified_number_of_nodes))

    else:
        # ploting only 1 column
        fig.suptitle('Dataset: {}'.format(data_set), fontsize=16)
        fig.suptitle('Dataset: {}, # of nodes: {}, # of layers: {} \n'.format(data_set, multilayer_graph.number_of_nodes, multilayer_graph.number_of_layers), fontsize=16)

        create_plot(multilayer_graph, axs)

    plt.savefig("figures/" + data_set + "_" + str(total_columns) + "_" + str(percentage) +".png", format="png")

    print(time.time()-start_time)

if __name__ == "__main__":
    main()