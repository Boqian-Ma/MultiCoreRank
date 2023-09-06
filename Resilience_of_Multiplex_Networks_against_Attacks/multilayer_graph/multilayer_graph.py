from ast import Raise
from os import error, getcwd
from os.path import dirname
from array import array
import gc
import itertools
import numpy as np
import scipy.stats
import networkx as nx

import matplotlib

# from Resilience_of_Multiplex_Networks_against_Attacks.core_decomposition.breadth_first_v3 import normalize
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        self.adjacency_list = None

        # Dataset source
        #self.dataset = dataset
        self.networkx_layers = [] # Network x layers implementations

        self.networkx_projection = None

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
        dataset_file = open(dirname(getcwd()) + '/Resilience_of_Multiplex_Networks_against_Attacks/used_clean_datasets/' + dataset_file + '.txt')
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
        for _, line in enumerate(dataset_file):
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
            # Only consider unweighted and undirected paths

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
        '''
        Only consider nodes that are active
        '''
        node_count = 0

        for i in range(1, self.number_of_nodes + 1):
            flag = False

            # if a node is active in a layer, we consider it as a node
            for j in range(self.number_of_layers):
                if len(self.adjacency_list[i][j]) > 0:
                    flag = True
            
            if flag is True:
                node_count += 1

        self.modified_number_of_nodes = node_count

    def keep_layers(self, layers_to_keep):
        '''
        Remove all layers besides the onces in layers_to_keep
        update information
        '''

        if len(layers_to_keep) > self.number_of_layers:
            print("too many layers to keep lmao")
            quit()

        new_adjacency_list = []

        new_adjacency_list = [[] for _ in self.nodes_iterator]

        for node in range(self.number_of_nodes + 1):

            for layer in range(self.number_of_layers):
                if layer in layers_to_keep:
                    # print(self.adjacency_list[node][layer])
                    new_adjacency_list[node].append(self.adjacency_list[node][layer])

        self.adjacency_list = new_adjacency_list
        self.layers_iterator = xrange(len(layers_to_keep) + 1)
        self.number_of_layers = len(layers_to_keep)


        print("Unmodified {}".format(self.modified_number_of_nodes))
        self._update_number_of_nodes()
        print("modified {}".format(self.modified_number_of_nodes))


        print(self.number_of_layers)

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

    def _load_networkx_projection(self):

        if self.dataset_file is not None:
            # Create base graph
            G = nx.Graph()
            # Add all nodes
            G.add_nodes_from([i for i in range(1, self.number_of_nodes + 1)])

            # loop through all layers and add an edge
            for layer in range(self.number_of_layers):
                # Add edges in this layer
                for from_node in range(1, len(self.adjacency_list)):
                    for to_node in self.adjacency_list[from_node][layer]:
                        if not G.has_edge(from_node, to_node):
                            G.add_edge(from_node, to_node)
            
            self.networkx_projection = G

            # Save graphs
            
            # nx.draw_networkx(self.networkx_projection)
            # plt.savefig("graph_figures/{}_projection.png".format(self.dataset_file), format="png")
            # plt.clf()

    def _load_networkx(self):
        '''
        Use the current adjancency list implement to create a list of 
        networkX objects. Each object represents a layer in the 
        multiplex network. 
        '''
        if self.dataset_file is not None and self.networkx_layers is not None:
            self.networkx_layers = [] # Reset to empty list
            # Create graph 
            for layer in range(self.number_of_layers):
                G = nx.Graph() # New graph
                # create nodes
                G.add_nodes_from([i for i in range(1, self.number_of_nodes + 1)])
                # Add edges in this layer
                for from_node in range(1, len(self.adjacency_list)):
                    for to_node in self.adjacency_list[from_node][layer]:
                        G.add_edge(from_node, to_node)
                self.networkx_layers.append(G)

            
        # Save graphs
        # for g in range(len(self.networkx)):
        #     nx.draw_networkx(self.networkx[g])
        #     plt.savefig("graph_figures/{}_{}.png".format(self.dataset_file, g), format="png")
        #     plt.clf()
    
    def eigenvector_centrality(self):
        '''
        Calculate eigen centrality of the network
        '''
        self._load_networkx()
        eigenvector_centrality_matrix = []

        for layer_graph in self.networkx_layers:
            # eigenvector_centrality_vector = nx.eigenvector_centrality(layer_graph, max_iter=1000)
            eigenvector_centrality_vector = nx.eigenvector_centrality_numpy(layer_graph)
            eigenvector_centrality_matrix.append(eigenvector_centrality_vector.values())
        
        # Take transpose
        eigenvector_centrality_matrix = np.asarray(eigenvector_centrality_matrix).T
        eigenvector_centrality = {}
        for node in range(len(eigenvector_centrality_matrix)):
            eigenvector_centrality[node + 1] = sum(eigenvector_centrality_matrix[node])

        return eigenvector_centrality
  
    def betweenness_centrality(self):
        '''
        Calculate betweenness centrality of each layer and aggregate
        '''
        self._load_networkx()

        betweeness_centrality_matrix = []

        for layer_graph in self.networkx_layers:
            # eigenvector_centrality_vector = nx.eigenvector_centrality(layer_graph, max_iter=1000)
            betweeness_centrality_vector = nx.betweenness_centrality(layer_graph, normalized=True)
            betweeness_centrality_matrix.append(betweeness_centrality_vector.values())
        
        # Take transpose
        betweeness_centrality_matrix = np.asarray(betweeness_centrality_matrix).T
        betweeness_centrality = {}

        for node in range(len(betweeness_centrality_matrix)):
            betweeness_centrality[node + 1] = sum(betweeness_centrality_matrix[node])

        return betweeness_centrality

    def closeness_centrality(self):
        '''
        Calculate betweenness centrality of each layer and aggregate
        '''
        self._load_networkx()

        closeness_centrality_matrix = []
        
        for layer_graph in self.networkx_layers:
            # eigenvector_centrality_vector = nx.eigenvector_centrality(layer_graph, max_iter=1000)
            closeness_centrality_vector = nx.closeness_centrality(layer_graph)
            closeness_centrality_matrix.append(closeness_centrality_vector.values())
        
        # Take transpose
        closeness_centrality_matrix = np.asarray(closeness_centrality_matrix).T
        closeness_centrality = {}

        for node in range(len(closeness_centrality_matrix)):
            closeness_centrality[node + 1] = sum(closeness_centrality_matrix[node])

        return closeness_centrality

    def betweenness_centrality_projection(self):
        self._load_networkx_projection()
        # normalized=True
        return nx.betweenness_centrality(self.networkx_projection, normalized=True)

    def closeness_centrality_projection(self):
        '''
        Calculate closeness centrality on projection network
        '''
        self._load_networkx_projection()
        return nx.closeness_centrality(self.networkx_projection)
    

    
    def overlap_degree_rank(self):
        '''
        Calculate overlapping degree of network

        Returns the influence
        '''
        map = {} # Result
        for node in range(1, self.number_of_nodes + 1):
            degree = 0
            for layer in self.adjacency_list[node]:
                degree += len(layer)
            map[node] = degree
    
        # Sort by influence
        # map = sorted(map.items(), key=operator.itemgetter(1), reverse=True)
        return map

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
            corr_matrix[layer1][layer2] = corr_matrix[layer2][layer1] = scipy.stats.pearsonr(x1, x2)[0]
            # will return nan results

        return corr_matrix

    def pearson_correlation_coefficient_find_negatives(self):
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


        disassortative_pairs = ["{}\n".format(self.dataset_file)]

        count = {}

        for layer1, layer2 in combinations:
            x1, x2 = self.get_layer_node_degrees(layer1, layer2)
            corr = corr_matrix[layer2][layer1] = scipy.stats.pearsonr(x1, x2)[0]

            # if two layers are disassortative
            if corr < 0:
                if "{} {} {}\n".format(layer2, layer1, corr) in disassortative_pairs:
                    continue
                
                disassortative_pairs.append("{} {} {}\n".format(layer1, layer2, corr))

                if layer1 in count:
                    count[layer1] += 1
                else: 
                    count[layer1] = 1
        
        # TODO: Clean up
        # sort
        # print(count)
        return disassortative_pairs, count

    def pearson_correlation_coefficient_find_positives(self):
        '''
        Compute pearson correlation of a graph
        '''
        layers = [x for x in range(self.number_of_layers)]

        # Get layers combination pairs
        combinations = list(itertools.permutations(layers, 2))

        corr_matrix = [[0] * self.number_of_layers for i in range(self.number_of_layers)]
        # initialise diaganol for heatmap
        for i in range(self.number_of_layers):
            corr_matrix[i][i] = 1


        # disassortative_pairs = ["{}\n".format(self.dataset_file)]
        disassortative_pairs = []

        count = {}

        for layer1, layer2 in combinations:
            x1, x2 = self.get_layer_node_degrees(layer1, layer2)
            corr = corr_matrix[layer2][layer1] = scipy.stats.pearsonr(x1, x2)[0]

            # if two layers are disassortative
            if corr > 0:

                if layer1 in count:
                    count[layer1] += 1
                else: 
                    count[layer1] = 1

                # if "{} {} {}\n".format(layer2, layer1, corr) in disassortative_pairs:
                #     continue
                
                # disassortative_pairs.append("{} {} {}\n".format(layer1, layer2, corr))
                disassortative_pairs.append((layer1, layer2, corr))
        
        disassortative_pairs = sorted(disassortative_pairs, key=lambda x: (-x[2], x[0]))  

        
        s = []

        for i in disassortative_pairs:
            s.append("{} {} {}\n".format(i[0], i[1], i[2]))

        # TODO: Clean up
        # sort
        # print(count)
        return s, count

    
   