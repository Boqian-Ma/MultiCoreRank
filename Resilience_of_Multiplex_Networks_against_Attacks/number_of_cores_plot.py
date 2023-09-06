import argparse
import math
from resource import error
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from multilayer_graph.multilayer_graph import MultilayerGraph
from core_decomposition.breadth_first_v3 import breadth_first as bfs
from utilities.print_file import PrintFile 

from os import getcwd
from os.path import dirname
from helpers import create_plot, create_plots, get_influence_node_ranking, get_influence, get_influence_node_tuples, get_influence_node_tuples_new

import pandas as pd

font = {'size': 18}

matplotlib.rc('font', **font)

def main(file_name):

    # retrive data

    # plot

    # set axis /home/z5260890/Resilience_of_Multiplex_Networks_against_Attacks/output/inner_most_cores_map/europe_0_100_1.txt
    # file_name = "europe_0_100_1"

    path = dirname(getcwd()) + "/output/inner_most_cores_map_fast/{}.txt".format(file_name)


    with open(path, "r") as file:
        data = file.read()

    x, y = process_data(data)

    return x, y

    


def process_data(s):
    l = s.split("\n")
    x = []
    y = []
    for item in l:
        a, _ , b = item.split(" ")
        x.append(int(a))
        y.append(int(b))

    return x, y

def plot(x, y, dataset, print_file):

    _, ax = plt.subplots(figsize=(8, 8))

    ax.step(x , y, '*', where='post', label=dataset, color="green")

    ax.grid(True)
    ax.legend(loc='center')
    # ax.set_title('Number of cores vs % of node removal in {}'.format(dataset))
    ax.set_xlabel('% of node removed')
    ax.set_ylabel('Number of cores')

    print_file.print_number_of_cores_plot(plt, dataset)

if __name__ == "__main__":

    # file_names = [("europe_0_100_1", "Europe"), 
    #                 ("northamerica_0_2_13_14_11_0_100_1", "North America"), 
    #                 ("southamerica_9_17_21_52_0_100_1", "South America"),
    #                 ("celegans_0_100_1", "Celegan"),
    #                 ("homo_0_100_1", "Homo"),
    #                 ("biogrid_0_100_1", "Biogrid")
    #             ]

    file_names = [(sys.argv[1], " ".join(sys.argv[2:]))]


    for pair in file_names:

        file_name = pair[0]
        dataset = pair[1]

        print_file = PrintFile(dataset)

        print(file_name)

        x, y = main(file_name)
        
        plot(x, y, dataset, print_file)

        
