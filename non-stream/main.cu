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
{
	float prefix_time,local_d_time,global_d_time;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int hash=1, cache=0, bitflag=1, NORMALIZE=1;
	#ifdef profile
	if(tid==0){
		printf("\n");
		for(int i=0; i<n_subgraph;i++)
		{
			S->candidate.vertices[i];
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
	// if(threadIdx.x==0){printf("warpID:%d, sourceIndex:%d,start: %d\n",warpId, sourceIndex, S->candidate.start[0]);}
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
		// if(warpTid==0){printf(" Source: %d, len: %d\n",source,NL);}
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

__global__ void
check_layer(Sampling *S, gpu_graph G,curandState *global_state,int n_subgraph, int FrontierSize, int NeighborSize, int Depth)
{
	float prefix_time,local_d_time,global_d_time;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int hash=1, cache=0, bitflag=1, NORMALIZE=1;
	int warpId = tid/32;

	// __syncwarp(); 
	
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
	// __syncwarp();

	#ifdef profile
	// if(threadIdx.x==0){printf("Block:%d, sourceIndex:%d, start: %d\n",blockIdx.x, sourceIndex, S->candidate.vertices[0]);}
	#endif
	S->candidate.end[0]= n_subgraph;
	// get degree for all frontiers
	int index= tid;
		// for(int i=0;i<(n_subgraph*FrontierSize);i++)
	// {
	// 	int vert= S->candidate.vertices[i];
	// 	S->frontier_degree[i] = G.degree_list[source];
	// }
	while(sourceIndex < n_subgraph)
	{
		int curr_depth=0;
		while(curr_depth<Depth)
		{
			
			// gather all frontiers
			int start_index = sourceIndex * FrontierSize;  
			int end_index = (sourceIndex+1) * FrontierSize;  
			#ifdef profile
			if(warpTid==0){printf("Warp: %d,SourceIdex: %d,Start: %d, End: %d\n",warpId,sourceIndex,start_index,end_index);}
			#endif
			int index= start_index+warpTid;
			for(index;index<end_index;index+=warpsize)
			{	
				int vert= S->candidate.vertices[index];
				int bias = VertexBias(vert, &G);
				S->wvar[warpId].degree[index-start_index]= (float)bias;
				#ifdef profile
				// printf("Vert: %d, Bias: %d\n",vert,bias);
				#endif
			}
			// __syncwarp();
			// pick one with ITS
			float r = curand_uniform(&local_state);
			int selectedIndex= ITS_MDRW(&S->wvar[warpId], local_state, &G, FrontierSize,r); 
			if(threadIdx.x==0){
				int selected = S->candidate.vertices[selectedIndex];
				#ifdef profile
				if(warpTid==0){printf("Random selected: %d, vertex: %d\n",selectedIndex, selected);}
				#endif
				int NL= G.degree_list[selected];
				if(NL==0){curr_depth+=1;continue;}
				// generate one random integer with range of (0,NL);
				int r=rand_integer(local_state,NL);
				int neighbor_start= G.beg_pos[selected];
				int sample= G.adj_list[r+neighbor_start] ;
				#ifdef profile
				if(warpTid==0){printf("NL: %d, New selected: %d, vertex: %d\n",NL,r, sample);}
				#endif
				int SampleID=sourceIndex;
				int pos=atomicAdd(&S->samples[SampleID].start[0],1);
				S->samples[SampleID].vertex[pos]=selected;
				S->samples[SampleID].edge[pos]=sample;
				if(warpTid==0){atomicAdd(&S->sampled_count[0],1);}
				// update the degree and frontier
				S->candidate.vertices[selectedIndex] = sample; 
				S->frontier_degree[selectedIndex] = G.degree_list[sample];
				#ifdef profile
				if(warpTid==0){printf("Next level. Curr Depth: %d\n",curr_depth);}
				#endif
			}
			// __syncwarp();
			curr_depth+=1;
		}
		if(warpTid==0){
			sourceIndex=atomicAdd(&S->candidate.start[0],1);
		}
		sourceIndex= __shfl_sync(0xffffffff,sourceIndex,0);
		// if(warpTid==0){printf("Next source. %d\n",sourceIndex);}
		// __syncwarp();
	}
	if(tid==0){printf("%d,",S->sampled_count[0]);}
}

struct arguments Sampler(char beg[100], char csr[100],int n_blocks, int n_threads, int n_subgraph, int FrontierSize, int NeighborSize, int Depth, struct arguments args, int rank)
{
	int *total=(int *)malloc(sizeof(int)*n_subgraph);
	int *host_counter=(int *)malloc(sizeof(int));
	int T_Group=n_threads/32;
   	int each_subgraph=Depth*NeighborSize;
    int total_length=each_subgraph*n_subgraph;
	int neighbor_length_max=n_blocks*6000*T_Group;
	int PER_BLOCK_WARP= T_Group;
	int BUCKET_SIZE=125;
	int BUCKETS=32;
	int warps = n_blocks * T_Group;

	int total_mem_for_hash=n_blocks*PER_BLOCK_WARP*BUCKETS*BUCKET_SIZE;	
	int total_mem_for_bitmap=n_blocks*PER_BLOCK_WARP*300;	
	//std::cout<<"Input: ./exe beg csr nblocks nthreads\n";
	int *bitmap, *node, *qstop_global, *qstart_global, *sample_id, *depth_tracker, *g_sub_index, *degree_l, *counter, *pre_counter; 	
	int *seeds=(int *)malloc(sizeof(int)*n_subgraph*FrontierSize);
	int *h_sample_id=(int *)malloc(sizeof(int)*n_subgraph*FrontierSize);
	int *h_depth_tracker=(int *)malloc(sizeof(int)*n_subgraph*FrontierSize);
	
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

	int deviceCount;
	HRR(cudaGetDeviceCount(&deviceCount));
	// printf("My rank: %d, totaldevice: %d\n", rank,deviceCount);
	// HRR(cudaSetDevice(rank%deviceCount));
	Sampling *sampler;
	Sampling S(ginst->edge_count, warps, 10000,n_subgraph, BUCKETS*BUCKET_SIZE, Depth*NeighborSize, FrontierSize, Depth);
	H_ERR(cudaMalloc((void **)&sampler, sizeof(Sampling)));

	for(int n=0;n<n_subgraph*FrontierSize;++n){
			seeds[n]= dis(gen);
			h_sample_id[n]=0;
			h_depth_tracker[n]=0;   
			// printf("%d\n",seeds[n]);
	}
	
	// else
	// {
	// 	for(int n=0;n<n_subgraph;n++){
	// 		for(int i=0;i<FrontierSize+1;i++){
	// 			seeds[i]= 2;
	// 		}
	// 		// HRR(cudaMemcpy(S.front[n].pool,seeds,sizeof(int)*FrontierSize, cudaMemcpyHostToDevice));
	// 	}
	// }

	HRR(cudaMemcpy(S.candidate.vertices,seeds,sizeof(int)*n_subgraph*FrontierSize, cudaMemcpyHostToDevice));
	HRR(cudaMemcpy(S.candidate.instance_ID,h_sample_id,sizeof(int)*n_subgraph*FrontierSize, cudaMemcpyHostToDevice));
	HRR(cudaMemcpy(S.candidate.depth,h_depth_tracker,sizeof(int)*n_subgraph*FrontierSize, cudaMemcpyHostToDevice));
	HRR(cudaMemcpy(sampler, &S, sizeof(Sampling), cudaMemcpyHostToDevice));
	// shared variable for bincount and tempQ
	double start_time,total_time;
	start_time= wtime();
	if(FrontierSize==1){
	check<<<n_blocks, n_threads>>>(sampler, ggraph, d_state, n_subgraph, FrontierSize, NeighborSize, Depth);
	}
	else{
		printf("Layer call\n");
		check_layer<<<n_blocks, n_threads>>>(sampler, ggraph, d_state, n_subgraph, FrontierSize, NeighborSize, Depth);
	}
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
	/*
	 	The sampled graph is stored as edge list. To get the samples in the CPU memory, copy each array from class Si to CPU allocated memory.
	 */
	// printf("Sampled edges:%d\n",host_counter[0]);	
	// args.sampled_edges=host_counter[0];
	args.time=total_time; 
	return args;
}
