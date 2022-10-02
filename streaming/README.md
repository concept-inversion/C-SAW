#### Streaming version:

##### Input format
Input: ./exe <dataset name> <beg file> <csr file> <ThreadBlocks> <Threads> <# of samples> <FrontierSize> <NeighborSize> <Depth/Length> <#GPUs>

Neighbor size represents how many neighbors to sample for each vertex.

##### Example:
mpirun -n 1 streaming.bin WG WG/beg.bin WG/csr.bin 10 128 1 40 5 3 1

or 

./run <# of samples> <FrontierSize> <NeighborSize> <Depth/Length> <#GPUs>  



##### Note:
The current source code only supports dividing the graph into four partition and streams two partitions into GPU. The dynamic version may be uploaded later. Memory allocation (espically for storing the samples and queue) may require higher allocation depending on the sampling parameters. 
Note, current version of this code works only with a single GPU. 
