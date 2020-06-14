#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "sampler.cuh"
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
using namespace std;

int main(int argc, char *argv[])
{
    if(argc!=11){std::cout<<"Input: ./exe <dataset name> <beg file> <csr file> <ThreadBlocks> <Threads> <# of samples> <FrontierSize> <NeighborSize> <Depth/Length> <#GPUs>\n";exit(0);}
    // SampleSize, FrontierSize, NeighborSize
    // printf("MPI started\n");
    int n_blocks = atoi(argv[4]);
    int n_threads = atoi(argv[5]);
    int SampleSize = atoi(argv[6]);
    int FrontierSize = atoi(argv[7]);
    int NeighborSize = atoi(argv[8]);
    int Depth= atoi(argv[9]);
    int total_GPU = atoi(argv[10]);
    
    MPI_Status status;
    int myrank;
    double global_max_time, global_min_time;
    int global_sampled_edges;
    struct arguments args;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
    int global_sum;
    SampleSize = SampleSize/total_GPU;
    
    args= Sampler(argv[2],argv[3], n_blocks, n_threads, SampleSize, FrontierSize, NeighborSize, Depth, args,myrank);
    MPI_Reduce(&args.time, &global_max_time, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&args.time, &global_min_time, 1, MPI_DOUBLE,MPI_MIN, 0, MPI_COMM_WORLD);
    float rate = global_sampled_edges/global_max_time/1000000;
    if(myrank==0)
    {
        printf("%s,%f,%f\n",argv[1],global_min_time,global_max_time);
    }
    MPI_Finalize();
   return 0;
}