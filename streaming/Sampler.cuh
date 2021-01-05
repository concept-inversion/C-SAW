#ifndef SAMP_H
#define SAMP_H
struct arguments{
    int sampled_edges;
    double time;
};

struct arguments Sampler(char beg[100],char csr[100], int n_blocks, int n_threads, int n_subgraph, int frontier_size, int neighbor_size,int depth,struct arguments args, int myrank);
#endif