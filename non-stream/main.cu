#include <iostream>
#include "graph.h"
#include "wtime.h"
#include <queue>
#include <set>
#include <iterator>
#include "gpu_graph.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "herror.h"
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include "api.cuh"
#include "sampler.cuh"
#include "sample_class.cuh"
#include "functions.cuh"
using namespace std;

__global__ void
check(Sampling *S, gpu_graph G,curandState *global_state,int n_subgraph, int FrontierSize, int NeighborSize, int Depth)
// check(graph,curandState *gs,sample,*neigh_l,n_blocks,*d_seed,n_threads,hash,bitmap)
{
	float prefix_time,local_d_time,global_d_time;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int hash=1, cache=0, bitflag=1, NORMALIZE=1;
	#ifdef profile
	if(tid==0){
		printf("Sources.\n");
		for(int i=0; i<n_subgraph;i++)
		{
			printf("%d\n",S->candidate.vertices[i]);
		}}
	#endif
	__syncwarp();
	int warpId = tid/32;
	int warpTid=threadIdx.x%32;
    clock_t start_time,stop_time;
	int i=0, warpsize=100;
	curandState local_state=global_state[threadIdx.x];
	curand_init(tid, 0, 0, &local_state); // sequence created with different seed and same sequence
	int __shared__ l_search[256];
	int __shared__ max_find[256];
	S->candidate.start[0] = 0;
	int sourceIndex=warpId,source=0;
	if(warpTid==0){
		atomicAdd(&S->candidate.start[0],1);
	}
	// sourceIndex= __shfl_sync(0xffffffff,sourceIndex,0);
	__syncwarp();

	#ifdef profile
	if(threadIdx.x==0){printf("Block:%d, sourceIndex:%d,start: %d\n",blockIdx.x, sourceIndex, S->candidate.start[0]);}
	#endif
	// start loop
	S->candidate.end[0]= n_subgraph;
	// clock_t start = clock();
	while(sourceIndex < S->candidate.end[0])
	{
		int VertCount=1;
		source= S->candidate.vertices[sourceIndex];
		int SampleID= S->candidate.instance_ID[sourceIndex];
		int NL= G.degree_list[source];
		#ifdef profile
		if(warpTid==0){printf(" Source: %d, len: %d\n",source,NL);}
		#endif
		if((NL==0) || (NL>8000)){	
			if(warpTid==0){sourceIndex=atomicAdd(&S->candidate.start[0],1);}
			sourceIndex= __shfl_sync(0xffffffff,sourceIndex,0);
			__syncwarp();
			continue;
		}
		int len= get_neighbors(&G,source,&S->wvar[warpId],VertCount);
			
			if(NORMALIZE==0){
				select(&S->wvar[warpId],&S->cache,NeighborSize,1,local_state, &G,S->count.colcount, source,S->max,bitflag,cache);
			}
			else{
				heur_normalize(&S->wvar[warpId],&S->cache,NeighborSize,1,local_state, &G,S->count.colcount, source, S->max,bitflag,cache);
				}
		frontier(&G,S,warpId,SampleID,NeighborSize,source,sourceIndex, hash, Depth);
		
		__syncwarp();
		 if(warpTid==0){
			sourceIndex=atomicAdd(&S->candidate.start[0],1);
		}
		sourceIndex= __shfl_sync(0xffffffff,sourceIndex,0);
		__syncwarp();
	}
	if(tid==0){printf("%d,",S->sampled_count[0]);}
}


struct arguments Sampler(char beg[100], char csr[100],int n_blocks, int n_threads, int n_subgraph, int FrontierSize, int NeighborSize, int Depth, struct arguments args, int rank)

{
	// printf("A: %d\n",args);
	// if(args!=9){std::cout<<"Wrong input\n"; return -1;
	// cout<<"\nblocks:"<<n_blocks<<"\tThreads:"<<n_threads<<"\tSubgraphs:"<<n_subgraph<<"\n";
	//int n_threads=32; 
	
	int *total=(int *)malloc(sizeof(int)*n_subgraph);
	int *host_counter=(int *)malloc(sizeof(int));
	int len=5;
	int T_Group=n_threads/32;
	int depth=10;	
   	int n_child=150;
   	int each_subgraph=depth*n_child;
    int total_length=each_subgraph*n_subgraph;
	int neighbor_length_max=n_blocks*6000*T_Group;
	int PER_BLOCK_WARP= T_Group;
	int BUCKET_SIZE=125;
	int BUCKETS=32;
	int warps = n_blocks * T_Group;

	int total_mem_for_hash=n_blocks*PER_BLOCK_WARP*BUCKETS*BUCKET_SIZE;	
	int total_mem_for_bitmap=n_blocks*PER_BLOCK_WARP*300;	
	//std::cout<<"Input: ./exe beg csr nblocks nthreads\n";
	
	const char *beg_file=beg;
	const char *csr_file=csr;
	const char *weight_file=csr;
	
	graph<long, long, long, vertex_t, index_t, weight_t>
	*ginst = new graph
	<long, long, long, vertex_t, index_t, weight_t>
	(beg_file,csr_file,weight_file);  
	gpu_graph ggraph(ginst);
	curandState *d_state;
	cudaMalloc(&d_state,sizeof(curandState));  

	Sampling *sampler;
	Sampling S(ginst->edge_count, warps, 10000,n_subgraph, BUCKETS*BUCKET_SIZE, BUCKETS, Depth);
	H_ERR(cudaMalloc((void **)&sampler, sizeof(Sampling)));

	// int *host_counter=(int *)malloc(sizeof(int));
	int *host_prefix_counter=(int *)malloc(sizeof(int));
    int *node_list=(int *)malloc(sizeof(int)*total_length);
    int *set_list=(int *)malloc(sizeof(int)*total_length);	
	
	int *degree_list=(int *)malloc(sizeof(int)*ginst->edge_count);
	std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1,10000);
	int numBlocks;
	//cudaGetDevice(&device);
    //cudaGetDeviceProperties(&prop, device);
    
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks,
    //     check,
    //     n_threads,
	// 	0);
	
	// cout<<"Max allocatable Blocks:"<<numBlocks<<"\n";

	int *d_node_list;
    int *d_edge_list;
	int *d_neigh_l;
	float *d_degree_l;   
	//float *d_random;
	int *d_seed;
	int *d_total; 

	int deviceCount;
	HRR(cudaGetDeviceCount(&deviceCount));
	// printf("My rank: %d, totaldevice: %d\n", rank,deviceCount);
	// HRR(cudaSetDevice(rank%deviceCount));
	int *bitmap, *node, *qstop_global, *qstart_global, *sample_id, *depth_tracker, *g_sub_index, *degree_l, *counter, *pre_counter;
	int *nodes;

	// HRR(cudaMalloc((void **) &hashtable,sizeof(int)*total_mem_for_hash));
	HRR(cudaMalloc((void **) &bitmap,sizeof(int)*total_mem_for_bitmap)); 	
	int *seeds=(int *)malloc(sizeof(int)*n_subgraph);
	int *h_sample_id=(int *)malloc(sizeof(int)*n_subgraph);
	int *h_depth_tracker=(int *)malloc(sizeof(int)*n_subgraph);
	
	for(int n=0;n<n_subgraph;++n)
    {
		seeds[n]=dis(gen);
		h_sample_id[n]=n;
		h_depth_tracker[n]=0;   
		// printf("%d\n",seeds[n]);
	}
    
	HRR(cudaMemcpy(S.candidate.vertices,seeds,sizeof(int)*n_subgraph, cudaMemcpyHostToDevice));
	HRR(cudaMemcpy(S.candidate.instance_ID,h_sample_id,sizeof(int)*n_subgraph, cudaMemcpyHostToDevice));
	HRR(cudaMemcpy(S.candidate.depth,h_depth_tracker,sizeof(int)*n_subgraph, cudaMemcpyHostToDevice));
	
	HRR(cudaMemcpy(sampler, &S, sizeof(Sampling), cudaMemcpyHostToDevice));
	// shared variable for bincount and tempQ
	double start_time,total_time;
	start_time= wtime();
	check<<<n_blocks, n_threads>>>(sampler, ggraph, d_state, n_subgraph, FrontierSize, NeighborSize, Depth);
	HRR(cudaDeviceSynchronize());
	// HRR(cudaMemcpy(host_counter, sampler->sampled_count, sizeof(int), cudaMemcpyDeviceToHost));	
	
	// int total_count=0;
	// for(int i=0; i < n_subgraph;i++){
	// 	int count= S.samples[i].start[0];
	// printf("Sampled: %d\n",host_counter[0]);
	// 	total_count+=count;
	// }
	total_time= wtime()-start_time;
	// printf("%s,SamplingTime:%.6f\n",argv[1],total_time);
	// Copy the sampled graph to CPU
	// printf("Sampled edges:%d\n",host_counter[0]);	
	// args.sampled_edges=host_counter[0];
	args.time=total_time; 
	return args;
}