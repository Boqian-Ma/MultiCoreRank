import argparse

from multilayer_graph.multilayer_graph import MultilayerGraph

from core_decomposition.breadth_first_v3 import breadth_first as bfs

from utilities.print_console import print_dataset_name, print_dataset_info, print_dataset_source
from utilities.print_file import PrintFile


if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='Core Decomposition and Densest Subgraph in Multilayer Networks')

    # arguments
    #parser.add_argument('n', help='dataset from which paper. "multilayer_layer_core_decomposition" or "measuring_and_modelling_correlations_in_multiplex_networks"')

    parser.add_argument('d', help='dataset')
    parser.add_argument('m', help='method')
    parser.add_argument('-b', help='beta', default=False, type=float)
    
    # options
    parser.add_argument('--ver', dest='ver', action='store_true', default=False ,help='verbose')
    parser.add_argument('--dis', dest='dis', action='store_true', default=False ,help='distinct cores')

    # read the arguments
    args = parser.parse_args()

    # create the input graph and print its name
    multilayer_graph = MultilayerGraph(args.d)
    print_dataset_name(args.d)
    #print_dataset_source(args.n)

    # create the output file if the --v option is provided
    if args.ver and args.m in {'bfs', 'dfs', 'h', 'n'}:
        print_file = PrintFile(args.d)
    # set it to None otherwise
    else:
        print_file = None

    # core decomposition algorithms
    if args.m == 'v3':
        print ('---------- Influence measure Version 3 ----------')
        ranking = bfs(multilayer_graph, print_file, args.dis, args.d)

    # dataset information
    elif args.m == 'info':
        print_dataset_info(multilayer_graph)

    # unknown input
    else:
        parser.print_help()

    

