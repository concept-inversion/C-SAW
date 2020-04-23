A Framework for Graph Sampling andRandom Walk on GPUs

We use CSR format of graph for sampling. Adjacency list of most datasets are available here. We use the code below to generate CSR from adjacency list. 
https://github.com/asherliu/graph_project_start

Generate the CSR and put the folder in main directory.

To run:

    Step 1: make

    Step 2: Define the required API in API.cuh
    
    Step 3: Update the dataset name in the run.sh file.

    Step 4: ./run.sh <Sample Size> <FrontierSize> <NeighborSize> 
