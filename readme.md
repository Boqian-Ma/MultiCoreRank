# Set up

python version: 2.75

1. Start new virtual environment
2. `pip install -r requirements.txt`
3. download `datasets` folder from teams (too big to upload onto github)
4. place `datasets` in root folder

# FOLDERS

- datasets: datasets listed in paper and more
- `Resilience_of_Multiplex_Networks_against_Attacks`: code
- `Resilience_of_Multiplex_Networks_against_Attacks/figures`: output plots
- `Resilience_of_Multiplex_Networks_against_Attacks/influence`: saved influence ranking
- `Resilience_of_Multiplex_Networks_against_Attacks/core_decomposition`: core decomposition algorithms
- `Resilience_of_Multiplex_Networks_against_Attacks/figures`: output plots
- output: destination of code's output (core decompositions)

# EXECUTION

Run the following command from the folder `Resilience_of_Multiplex_Networks_against_Attacks/`:

`python main.py d m p c`

Examples

`python main.py example o 0.2 5` (iteratively removing top 20% of influencial nodes and display 4 iterations (first column is full network))

## positional arguments:

Dataset "d"

smaller datasets:

- example
- aarhus
- biogrid
- celegans
- europe
- asia
- sacchcere
- northamerica
- oceania
- pierreauger_multiplex
- southamerica

Large dataset

- homo
- dblp
- obamainisrael
- amazon
- friendfeedtwitter
- higgs
- friendfeed

Method "m"

- i (iterative: calculate node influence before node removal in each iteration)
- o (once off: only calculate/load influence at the beginning)

Percentage "p":

- 0 < p < 1 (percentage of node removal)

Columns "c":

- 1 <= p <= 5 (number of columns displayed on final output plots or number of iterations to remove the percentage of nodes specified in "p")

---

Katana notes

Manage jobs: https://unsw-restech.github.io/using_katana/running_jobs.html#managing-jobs-on-katana
Show all jobs in system
qstat | less
Show my own job
qstat -u $USER
qstat -su $USER
qstat -f <job id>

show info about a job
qstat -f -x 2067936

#PBS -l select=1:ncpus=8:mem=124gb
#PBS -l walltime=12:00:00
#PBS -M boqian.ma@student.unsw.edu.au

CORE DECOMPOSITION AND DENSEST SUBGRAPH IN MULTILAYER NETWORKS

FOLDERS

- datasets: datasets listed in Table 1 and example network of Figure 1
- multilayer_core_decomposition: code
- output: destination of code's output

CODE
To use the code, first run 'python setup.py build_ext --inplace' from the folder 'multilayer_core_decomposition/'.
This command builds the .c files created by Cython.
Alternatively, without running the mentioned command, it is possible to directly execute the Python code.

EXECUTION
Run the following command from the folder 'multilayer_core_decomposition/':
'python multilayer_core_decomposition.py [-h] [-b B] [--ver] [--dis] d m'

positional arguments:

- d dataset
  - example
  - homo
  - sacchcere
  - dblp
  - obamainisrael
  - amazon
  - friendfeedtwitter
  - higgs
  - friendfeed
- m method
  - n naive method (beginning of Section 3)
  - bfs BFS-ML-cores (Algorithm 2)
  - dfs DFS-ML-cores, (Algorithm 3)
  - h HYBRID-ML-cores, (Algorithm 4)
  - ds ML-densest (Algorithm 5)
  - info dataset info

optional arguments:

- -h, --help show the help message and exit

- -b B beta
  required for ML-densest (Algorithm 5)
- --ver verbose
  print the resulting multilayer core decomposition in the output folder with the format 'coreness_vector size nodes'
- --dis distinct cores
  filter distinct cores removing duplicates (please note that this option requires additional memory)

example:
'python multilayer_core_decomposition.py homo h --ver'

SCRIPT
The same result obtained by option '--dis' can be achieved by executing a multilayer core decomposition method with option '--ver' and then running the following command from the folder 'multilayer_core_decomposition/scripts/':
'python filter_distinct_cores.py [-h] d'

positional arguments:

- d dataset

optional arguments:

- -h, --help show the help message and exit

example:
'python filter_distinct_cores.py homo'

# datasets from external parties

WARNING!!!! Each data set is provided "AS IS", without any implied warranty of suitability for any particular use.
The data sets are made available for research purposes only.
If you use a data set in your research, please remember to include a reference to the relevant paper, as specified in each data set record.
C. Elegans neural network
Description:
The neural network of the C.elegans nematode worm. The two (undirected) layers represent, respectively, synapses and gap junctions.
Nodes: 281
Layers: 2
Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
Dowload: celegans.tar.gz (6.1 KB) (md5sum: bb4d97e3d74b171c095f1bced47335d2)

    BIOGRID gene-protein interaction network
    Description:
    The network of physical and genetic interactions among all proteins in the BIOGRID data set.
    Nodes: 54549
    Layers: 2
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: BIOGRID.tar.gz (1.6 MB) (md5sum: 918eee4dcd9029b2be1d44033efdcf5c)


    OpenFlights continental airport networks
    Description:
    This is a set of six continental multiplex network of air transport (Africa, Asia, Europe, North America, Oceania, South America). Each layer represents an airline, and the edges indicate the presence of a direct flight between the two corresponding airports.

    Africa
    Nodes: 235
    Layers: 84
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: layers_Africa.tar.gz (5.9 KB) (md5sum: c0437f695391b8aceff6a1725b8910e8)

    Asia
    Nodes: 792
    Layers: 213
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: layers_Asia.tar.gz (51 KB) (md5sum: 4603149a325deaa8719125781eab7d8c)

    Europe
    Nodes: 593
    Layers: 175
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: layers_Europe.tar.gz (38 KB) (md5sum: a36046bd242535982edfadb4990d35cb)

    North America
    Nodes: 1020
    Layers: 143
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: layers_NorthAmerica.tar.gz (42 KB) (md5sum: 63d3deff1f8f16ce1e304c8e541a4c1c)

    South America
    Nodes: 296
    Layers: 58
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: layers_SouthAmerica.tar.gz (7.1 KB) (md5sum: 3fe66c503bce05e8c94a662d87a6e4d8)

    Oceania
    Nodes: 261
    Layers: 37
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: layers_Oceania.tar.gz (4.6 KB) (md5sum: a86030a0279dd3a21575f565d5e4af3d)

    APS Scientific Collaboration Network
    Description:
    The network of scientific collaboration among all the authors who have published at least one paper in any journal of the Americal Physical Society (APS). Each (unweighted and undirected) layer corresponds to the collaboration network in one of the ten highest-level categories (0-9) in the Physics and Astronomy Classification Scheme (PACS). Two authors are linked at one layer if they have co-authored at least one paper with a PACS code in that area.
    Nodes: 170397
    Layers: 10
    Rerefence: V. Nicosia, V. Latora "Measuring and modelling correlations in multiplex networks", Phys. Rev. E 92, 032805 (2015) (Abstract - APS)
    Dowload: APS.tar.gz (47 MB) (md5sum: e551647048ae8035bb27b70788a15c94)


Datasets
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GSOPCK