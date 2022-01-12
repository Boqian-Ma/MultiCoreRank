import math
from multilayer_graph.multilayer_graph import MultilayerGraph
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_console import print_dataset_name, print_dataset_info, print_dataset_source
from utilities.print_file import PrintFile 

from scipy.interpolate import spline

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

    # normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    
    print(pearson_flat_list)

    hist = axs[1, plot_col].hist(pearson_flat_list, bins=10, weights=weights)

    #hist = axs[1, plot_col].hist(pearson_flat_list, weights=weights)
    axs[1, plot_col].set_xlim(-1, 1.5)
    axs[1, plot_col].set_ylabel('Density')
    axs[1, plot_col].set_xlabel('Pearson Correlation Coefficient')

    # pearson density function x axis limit
    '''
    density, bins, _ = hist
    bin_centers = 0.5 * (bins[1: ] + bins[ :-1])
    xnew = np.linspace(-1, 1, 1000)  

    density_smooth = spline(bin_centers, density, xnew, order=3)
    axs[1, plot_col].plot(xnew, density_smooth)

    axs[1, plot_col].set_xlim(-1, 1)

    # set y limit
    density_y_lim = axs[1, plot_col].get_ylim()
    axs[1, plot_col].set_ylim(0, density_y_lim[1])
    '''
    return None

def create_plot(multilayer_graph, axs):

    pearson_coe_matrix = multilayer_graph.pearson_correlation_coefficient()
    pearson_flat_list = [item for sublist in pearson_coe_matrix for item in sublist]
    
    # print(len(pearson_flat_list))    

    # heat map
    im = axs[0].imshow(pearson_coe_matrix, cmap='Greens', origin='lower', interpolation='none')
    im.set_clim(-1, 1)    

    plt.colorbar(im, ax=axs[0])

    # Normalised histogram
    weights = np.ones_like(pearson_flat_list) / float(len(pearson_flat_list))   # calculate weights
    hist = axs[1].hist(pearson_flat_list, bins=10, weights=weights)

    #axs[2].hist(pearson_flat_list, bins=100, weights=weights, histtype="step")
    #sns.distplot(pearson_flat_list, ax=axs[2], hist_kws={'weights': weights},  kde=False)

    axs[1].set_xlim(-1, 1)
    axs[1].set_ylabel('Density')
    axs[1].set_xlabel('Pearson Correlation Coefficient')

    # Plot 2
    density, bins, _ = hist
    bin_centers = 0.5 * (bins[1: ] + bins[ :-1])

    print(bin_centers)

    # axs[2].plot(bin_centers, density) 
    axs[2].set_xlim(-1, 1)
    axs[2].set_ylabel('Density')
    axs[2].set_xlabel('Pearson Correlation Coefficient')

    xnew = np.linspace(-1, 1, 1000)  
    density_smooth = spline(bin_centers, density, xnew)
    axs[2].plot(xnew, density_smooth)
    # sns.kdeplot(bin_centers, ax=axs[2])
    # Check area underneith
    # widths = bins[1:] - bins[:-1]
    #assert((density * widths).sum() == 1.0)
    # print((density * widths).sum())

def new_axis(arr):
    res = []
    for i in arr:
        res.append([i])
    return np.array(res)

def main():
    start_time = time.time()
    data_set = "example"
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
        create_plot(multilayer_graph, axs)

    plt.savefig("figures/" + data_set + "_" + str(total_columns) + "_" + str(percentage) +".png", format="png")

    print(time.time()-start_time)
    quit()

if __name__ == "__main__":
    main()