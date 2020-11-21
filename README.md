<!-- start: header -->
<p align="center">
  <a href="https://github.com/concept-inversion/C-SAW"><img src="/images/C-SAW1_modified.png"></a>
</p>
<!-- end: header -->

#### C-SAW: A Framework for Graph Sampling and Random Walk on GPUs
---
C-SAW is a GPU based framework which can be used to implement variants of graph sampling and random walk algorithms. 

This repo contains two folders. One for streaming sampling for large graph and another for non-streaming sampling for graphs that fit in GPU memory. 


C-SAW uses CSR format of graph for sampling. Web-google dataset is included in the repo as example. Adjacency list of most datasets are available here.
http://snap.stanford.edu/data/index.html

The adjacency list can be converted into CSR using this library:
https://github.com/asherliu/graph_project_start



Generate the CSR and put the folder in main directory of both non-streaming and streaming sampling.

To run:
   
    Step 1: Define the required API in API.cuh inside the non-streaming folder.
 
    Step 2: Go to streaming or non streaming folder. Run make command.
    
    Step 3: Update the dataset name in the run.sh file.

    Step 4: ./run.sh <# of samples> <FrontierSize> <NeighborSize> <Depth/Length> <nGPUs> 

For changing the depth of the sampling or length of the random walk, update the DEPTH_LIMIT within Sampling class in sample_class.cuh at non-stream folder. You can also change the memory allocation and other paramters with the Sampling class.

The sampled graph is stored as edge list in the GPU memory as a class variable Si found in sample_class.cuh. The output format:
```
Edges sampled, dataset name, min-time, max-time
```

`min-time` and `max-time` is same for single GPU. SEPS can be computed as `Edges sampled/max-time`.


For more details, please refer to our [paper](https://arxiv.org/abs/2009.09103).

Citation:

```
@INPROCEEDINGS {,
author = {S. Pandey and L. Li and A. Hoisie and X. Li and H. Liu},
booktitle = {2020 SC20: International Conference for High Performance Computing, Networking, Storage and Analysis (SC)},
title = {C-SAW: A Framework for Graph Sampling and Random Walk on GPUs},
year = {2020},
volume = {},
issn = {},
pages = {780-794},
keywords = {},
doi = {10.1109/SC41405.2020.00060},
url = {https://doi.ieeecomputersociety.org/10.1109/SC41405.2020.00060},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {nov}
}
```
