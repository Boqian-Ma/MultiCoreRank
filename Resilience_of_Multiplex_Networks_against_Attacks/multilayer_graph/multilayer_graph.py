from os import error, getcwd
from os.path import dirname
from array import array
import gc

import itertools

import numpy as np
import scipy.stats

class MultilayerGraph:
    # def __init__(self, dataset_path=None, dataset='multilayer_layer_core_decomposition'):
    def __init__(self, dataset_file=None):
        # ****** instance variables ******
        # layers
        self.number_of_layers = 0
        self.layers_iterator = xrange(0)
        self.layers_map = {}

        # nodes and adjacency list
        self.number_of_nodes = 0
        self.modified_number_of_nodes = 0

        self.maximum_node = 0
        self.nodes_iterator = xrange(0)
        self.adjacency_list = []

        # Dataset source
        #self.dataset = dataset

        # if dataset_path has been specified
        if dataset_file is not None:
            # read the graph from the specified path
            self.load_dataset(dataset_file)
            # set the dataset path
            self.dataset_file = dataset_file
                
        # call the garbage collector
        gc.collect()

    def load_dataset(self, dataset_file):
        # open the file
        dataset_file = open(dirname(getcwd()) + '/datasets/' + dataset_file + '.txt')
        # read the first line of the file
        first_line = dataset_file.readline()
        split_first_line = first_line.split(' ')

        # set the number of layers
        self.number_of_layers = int(split_first_line[0])
        self.layers_iterator = xrange(self.number_of_layers)
        # set the number of nodes
        self.number_of_nodes = int(split_first_line[1])
        self.modified_number_of_nodes = int(split_first_line[1])
        self.maximum_node = int(split_first_line[2])
        self.nodes_iterator = xrange(self.maximum_node + 1)

        # create the empty adjacency list
        self.adjacency_list = [[array('i') for _ in self.layers_iterator] for _ in self.nodes_iterator]

        # map and oracle of the layers
        layers_map = {}
        layers_oracle = 0

        # for each line of the file
        for index, line in enumerate(dataset_file):
            # split the line
            split_line = line.split(' ')
            layer = int(split_line[0])
            from_node = int(split_line[1])
            to_node = int(split_line[2])

            # if the layer is not in the map
            if layer not in layers_map:
                # add the mapping of the layer
                layers_map[layer] = layers_oracle
                self.layers_map[layers_oracle] = layer
                # increment the oracle
                layers_oracle += 1

            # add the undirected edge
            self.add_edge(from_node, to_node, layers_map[layer])

    def add_edge(self, from_node, to_node, layer):
        # if the edge is not a self-loop
        '''
        TODO: what if there are duplicates
        i.e. the file has the following lines
            1 2 18
            1 18 2
        Only one line should be recorded, otherwise the number of links will double
        '''
        if from_node != to_node:
            # add the edge
            # Check if nodes are already linked to prevent duplicates
 
            # self.adjacency_list[from_node][layer].append(to_node)
            # self.adjacency_list[to_node][layer].append(from_node)

            if to_node not in self.adjacency_list[from_node][layer] and from_node not in self.adjacency_list[to_node][layer]:
                self.adjacency_list[from_node][layer].append(to_node)
                self.adjacency_list[to_node][layer].append(from_node)
            # else:
            #     print("yeet")

            
    # ****** Remove a node from the graph ******
    def _remove_node(self, node):
        '''
            Remove a node from the graph and unlink all edgesadobe
        '''
        for i in range(self.number_of_layers):
            # clear connections of given node
            self.adjacency_list[node][i] = array('i') 

        # Remove node from all other adjucency list elements
        for i in range(1, self.number_of_nodes + 1):
            for j in range(self.number_of_layers):
                if node in self.adjacency_list[i][j]:
                    self.adjacency_list[i][j].remove(node)

        # Update total nodes

    # ****** update total number of nodes after node removal ******
    def _update_number_of_nodes(self):
        node_count = 0

        for i in range(1, self.number_of_nodes + 1):
            flag = False

            for j in range(self.number_of_layers):
                if len(self.adjacency_list[i][j]) > 0:
                    flag = True

            if flag is True:
                node_count += 1

        print(node_count)

        self.modified_number_of_nodes = node_count

       

    # ****** Remove a list of node from the graph ******
    def remove_nodes(self, nodes):
        for node in nodes:
            self._remove_node(node)
        self._update_number_of_nodes()

    # ****** nodes ******
    def get_nodes(self):
        '''
        Get the set of connected nodes, excluding nodes with no connections
        '''  
        if self.number_of_nodes == self.maximum_node:
            nodes = list(set(self.nodes_iterator))
            nodes.remove(0)

            # Flag to see if node is isolated
            for node in range(1, len(nodes) + 1):

                flag = False    # track of the node is an orphan

                for layer in range(self.number_of_layers):
                    if len(self.adjacency_list[node][layer]) != 0:
                        flag = True
                        break
                
                # if all layers are empty
                if flag == False:
                    nodes.remove(node)  
        
            return set(nodes)

        else:
            # Flag to see if node is isolated
            nodes = list(set(self.nodes_iterator))
            for node in range(1, len(nodes) + 1):

                flag = False    # track of the node is an orphan

                for layer in range(self.number_of_layers):
                    if len(self.adjacency_list[node][layer]) != 0:
                        flag = True
                        break
                
                # if all layers are empty
                if flag == False:
                    nodes.remove(node)  
            return set(nodes)

    # ****** edges ******
    def get_number_of_edges(self, layer=None):
        number_of_edges = 0

        for neighbors in self.adjacency_list:
            for inner_layer, layer_neighbors in enumerate(neighbors):
                if layer is None:
                    number_of_edges += len(layer_neighbors)
                elif layer == inner_layer:
                    number_of_edges += len(layer_neighbors)

        return number_of_edges / 2

    def get_number_of_edges_layer_by_layer(self):
        number_of_edges_layer_by_layer = {}
        for layer in self.layers_iterator:
            number_of_edges_layer_by_layer[layer] = sum([len(neighbors[layer]) for neighbors in self.adjacency_list]) / 2
        return number_of_edges_layer_by_layer

    # ****** layers ******
    def get_layer_mapping(self, layer):
        return self.layers_map[layer]


    def get_layer_node_degrees(self, layer1, layer2):
        '''
        Given two layers return two lists of node degrees for all nodes 
        Used for pearson coefficient
        '''
        degrees1 = []
        degrees2 = []

        # first node is used for a placeholder, always empty
        for node in range(1, self.number_of_nodes + 1):
            degrees1.append(len(self.adjacency_list[node][layer1]))
            degrees2.append(len(self.adjacency_list[node][layer2]))
        
        #print(degrees1, degrees2)
        return degrees1, degrees2
    
    def pearson_correlation_coefficient(self):
        '''
        Compute pearson correlation of a graph
        '''
        layers = [x for x in range(self.number_of_layers)]

        # Get layers combination pairs
        combinations = list(itertools.combinations(layers, 2))

        corr_matrix = [[0] * self.number_of_layers for i in range(self.number_of_layers)]

        # initialise diaganol for heatmap
        for i in range(self.number_of_layers):
            corr_matrix[i][i] = 1

        for layer1, layer2 in combinations:
            x1, x2 = self.get_layer_node_degrees(layer1, layer2)

            #print(x1, x2)

            corr_matrix[layer1][layer2] = corr_matrix[layer2][layer1] = scipy.stats.pearsonr(x1, x2)[0]

            # will return nan results

        return corr_matrix


    def speaman_rank_correlation_coefficient(self):
        pass    
    