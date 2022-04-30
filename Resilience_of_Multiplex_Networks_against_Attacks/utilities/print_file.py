import errno
from os import getcwd
from os.path import dirname
import os
import json

class PrintFile:

    def __init__(self, dataset_path=None):
        # create the output file
        self.data_path = dataset_path

        self._create_file(dirname(getcwd()) + '/output/full_core_decomposition/' + dataset_path + '_core_decomposition_full.txt')
        self.full_core_decomposition_file = open(dirname(getcwd()) + '/output/full_core_decomposition/' + dataset_path + '_core_decomposition_full.txt', 'w+')
        
        self.partial_core_decomposition_file = dirname(getcwd()) + '/output/partial_core_decomposition/' + dataset_path + '_core_decomposition_partial.txt'   
        self.full_influence_rank_file = dirname(getcwd()) + '/output/full_influence_ranking/' + dataset_path + '_influence_ranking_full.txt'
        
        self.full_influence_rank_file_new = dirname(getcwd()) + '/output/full_influence_ranking_new/' + dataset_path + '_influence_ranking_full.txt'

        
        self.partial_influence_rank_file = dirname(getcwd()) + '/output/partial_influence_ranking/' + dataset_path + '_influence_ranking_partial.txt'
        self.figure_file = None
        self.correlation_file = dirname(getcwd()) + '/output/correlation/' + dataset_path + "_correlation.txt"

        # Create files
        # self._create_file(self.full_core_decomposition_file)
        # self._create_file(self.partial_core_decomposition_file)
        self._create_file(self.full_influence_rank_file)
        self._create_file(self.full_influence_rank_file_new)
        self._create_file(self.partial_influence_rank_file)
        self._create_file(self.correlation_file)

    

    def _create_file(self, file_name):
        '''
        Create a file to store data
        '''
        if not os.path.exists(os.path.dirname(file_name)):
            try:
                os.makedirs(os.path.dirname(file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    # def print_rank_correlation(self, string, percentage, columns):

    #     full_path = dirname(getcwd()) + "/output/rank_correlation/{}_{}_{}.png".format(self.data_path, percentage, columns)
    #     self._create_file(full_path)

    #     with open(self.correlation_file, 'w+') as f:
    #         f.writelines(correlation)

    #     pass


    def print_core(self, vector, k_core):
        # sort the nodes of the core
        sorted_k_core = list(k_core)
        sorted_k_core.sort()
        # write the core to the output file
        self.full_core_decomposition_file.write(str(vector) + '\t' + str(len(sorted_k_core)) + '\t' + str(sorted_k_core).replace('[', '').replace(']','') + '\n')

    def print_full_influence_rank(self, influence_rank):

        with open(self.full_influence_rank_file, 'w+') as f:
            for i in influence_rank:
                f.write(str(i[0]) + "\t" + str(i[1]) + "\n")

    def print_full_influence_rank_new(self, influence_rank):

        with open(self.full_influence_rank_file_new, 'w+') as f:
            for i in influence_rank:
                f.write(str(i[0]) + "\t" + str(i[1]) + "\n")


    def print_partial_influence_rank(self, influence_rank):
        with open(self.partial_influence_rank_file, 'w+') as f:
            for i in influence_rank:
                f.write(str(i[0]) + "\t" + str(i[1]) + "\n")


    def print_influence_distribution(self, plt, figure_name):
        '''
        print random figures
        '''
        full_path = dirname(getcwd()) + "/output/figures/influence_distribution/{}.png".format(figure_name)
        self._create_file(full_path)
        plt.savefig(full_path, format="png")


    def print_figure(self, plt, total_columns, percentage, iterative=False, flag=False):
        '''
        Save plt heatmap and histogram
        '''
        if iterative:
            full_path = dirname(getcwd()) + "/output/figures/{}_{}_{}_iterative.png".format(self.data_path, total_columns, percentage)
        else:
            full_path = dirname(getcwd()) + "/output/figures/{}_{}_{}_once.png".format(self.data_path, total_columns, percentage)
        
        if flag:
            full_path = dirname(getcwd()) + "/output/figures/assortatvity/{}_{}_{}.png".format(self.data_path, total_columns, percentage)

        self._create_file(full_path)
        plt.savefig(full_path, format="png")

    def print_subfigure(self, plt, layers_to_keep):
        '''
        Save plt heatmap and histogram
        '''
        if layers_to_keep:
            layers_to_keep = [str(x) for x in layers_to_keep]
            layer_string = "_".join(layers_to_keep)
            full_path = dirname(getcwd()) + "/output/figures/{}_{}.png".format(self.data_path, layer_string)
        else:
            full_path = dirname(getcwd()) + "/output/figures/{}_full_graph.png".format(self.data_path)

        
        self._create_file(full_path)
        plt.savefig(full_path, format="png")

    def print_correlation_heatmap(self, plt):
        '''
        Save rank method correlation heatmap
        '''
        full_path = dirname(getcwd()) + "/output/figures/rank_heatmap/{}_rank_method_correlation.png".format(self.data_path)
        self._create_file(full_path)
        plt.savefig(full_path, format="png")


    def print_correlation(self, correlation):
        '''
        Write correlation file
        '''

        full_path = dirname(getcwd()) + "/output/average_layers_correlation/{}_ave_corr.txt".format(self.data_path)
        self._create_file(full_path)

        with open(self.correlation_file, 'w+') as f:
            f.writelines(correlation)
    
    def print_correlation_new(self, correlation):
        '''
        Write correlation of centralitie methods file
        '''
        full_path = dirname(getcwd()) + '/output/correlation_new/' + self.data_path + "_correlation.txt"
        self._create_file(full_path)

        with open(full_path, 'w+') as f:
            f.writelines(correlation)

    def print_count_dis_layers(self, count):
        full_path = dirname(getcwd()) + "/output/disassortative_layers/{}_dis_layers_count.txt".format(self.data_path)
        self._create_file(full_path)

        with open(full_path, 'w+') as f:
            f.writelines(count)         

    def print_negative_correlation_layers(self, layers):
        '''
        print layer pair with negative correlation
        '''
        full_path = dirname(getcwd()) + "/output/disassortative_layers/{}_dis_layers.txt".format(self.data_path)
        self._create_file(full_path)

        with open(full_path, 'w+') as f:
            f.writelines(layers)        

    def print_average_correlation_layers(self, dictionary):
        '''
        print average layerwise correlation for each layer
        '''
        full_path = dirname(getcwd()) + "/output/average_layers_correlation/{}_ave_corr.txt".format(self.data_path)
        self._create_file(full_path)

        with open(full_path, 'w+') as f:
            json.dump(dictionary, f, indent=4)
