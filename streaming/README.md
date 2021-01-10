#### Streaming version:

##### Input format
Input: ./exe <dataset name> <beg file> <csr file> <ThreadBlocks> <Threads> <# of samples> <FrontierSize> <NeighborSize> <Depth/Length> <#GPUs>

##### Example:
./streaming.bin WG WG/beg.bin WG/csr.bin 100 128 400 10 5 5 1

or 

./run <# of samples> <FrontierSize> <NeighborSize> <Depth/Length> <#GPUs>  



##### Note:
The current source code only supports dividing the graph into four partition and streams two partitions into GPU. The dynamic version may be uploaded later. 
