A Framework for Graph Sampling andRandom Walk on GPUs

This repo contains two folders. One for streaming sampling for large graph and another for non-streaming sampling for graphs that fit in GPU memory.

We use CSR format of graph for sampling. Adjacency list of most datasets are available here. We use the code below to generate CSR from adjacency list. 
https://github.com/asherliu/graph_project_start

Web-google dataset is included in the repo as example. 

Generate the CSR and put the folder in main directory of both non-streaming and streaming sampling.

To run:
   
    Step 1: Define the required API in API.cuh inside the non-streaming folder.
 
    Step 2: Go to streaming or non streaming folder. Run make command.
    
    Step 3: Update the dataset name in the run.sh file.

    Step 4: ./run.sh <Sample Size> <FrontierSize> <NeighborSize> 

For changing the depth of the sampling or length of the random walk, update the DEPTH_LIMIT within Sampling class in sampler.cuh at non-stream folder. You can also change the memory allocation and other paramters with the Sampling class.
