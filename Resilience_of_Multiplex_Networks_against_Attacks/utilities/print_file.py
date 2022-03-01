import errno
from os import getcwd
from os.path import dirname
import os

class PrintFile:

    def __init__(self, dataset_path=None):
        # create the output file
        self.data_path = dataset_path

        self._create_file(dirname(getcwd()) + '/output/full_core_decomposition/' + dataset_path + '_core_decomposition_full.txt')
        self.full_core_decomposition_file = open(dirname(getcwd()) + '/output/full_core_decomposition/' + dataset_path + '_core_decomposition_full.txt', 'w+')
        
        self.partial_core_decomposition_file = dirname(getcwd()) + '/output/partial_core_decomposition/' + dataset_path + '_core_decomposition_partial.txt'   
        self.full_influence_rank_file = dirname(getcwd()) + '/output/full_influence_ranking/' + dataset_path + '_influence_ranking_full.txt'
        self.partial_influence_rank_file = dirname(getcwd()) + '/output/partial_influence_ranking/' + dataset_path + '_influence_ranking_partial.txt'

        self.figure_file = None

        self.correlation_file = dirname(getcwd()) + '/output/correlation/' + dataset_path + "_correlation.txt"

        # Create files
        # self._create_file(self.full_core_decomposition_file)
        # self._create_file(self.partial_core_decomposition_file)
        self._create_file(self.full_influence_rank_file)
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

    def print_partial_influence_rank(self, influence_rank):
        with open(self.partial_influence_rank_file, 'w+') as f:
            for i in influence_rank:
                f.write(str(i[0]) + "\t" + str(i[1]) + "\n")

    def print_figure(self, plt, total_columns, percentage, iterative=False):
        '''
        Save plt heatmap and histogram
        '''
        if iterative:
            full_path = dirname(getcwd()) + "/output/figures/{}_{}_{}_iterative.png".format(self.data_path, total_columns, percentage)
        else:
            full_path = dirname(getcwd()) + "/output/figures/{}_{}_{}_once.png".format(self.data_path, total_columns, percentage)
        self._create_file(full_path)
        plt.savefig(full_path, format="png")

    def print_correlation(self, correlation):
        '''
        Write correlation file
        '''
        with open(self.correlation_file, 'w+') as f:
            f.writelines(correlation)