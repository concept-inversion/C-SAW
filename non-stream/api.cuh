#ifndef API_H
#define API_H

__device__ int 
VertexBias(int vertexID, gpu_graph *graph)
{ 
    // For MDRW
    return graph->degree_list[vertexID];
    // For other
    // return 1;
}

__device__ int
EdgeBias(int vertexID, gpu_graph *graph)
{
    // For BNS, LS
    return graph->degree_list[vertexID];
    // For BRW, 
    // return 1;
}


__device__ int
Update(gpu_graph *G, int selected, int source)
{
    return selected;
}


#endif
