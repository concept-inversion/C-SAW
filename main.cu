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
#include "functions.cuh"
using namespace std;


__global__ void
check(Sampling *S, gpu_graph G,curandState *global_state,int n_subgraph, int N, int overN, int NORMALIZE, int bias, int bitflag, int hash, int cache)
// check(graph,curandState *gs,sample,*neigh_l,n_blocks,*d_seed,n_threads,hash,bitmap)
{
	float prefix_time,local_d_time,global_d_time;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpId = tid/32;
	int warpTid=threadIdx.x%32;
    clock_t start_time,stop_time;
	int i=0, warpsize=100;
	curandState local_state=global_state[threadIdx.x];
	curand_init(tid, 0, 0, &local_state); // sequence created with different seed and same sequence
	int __shared__ l_search[256];
	int __shared__ max_find[256];
	S->candidate.start[0] = 0;
	int sourceIndex=0,source=0;
	if(warpTid==0){
		sourceIndex=atomicAdd(&S->candidate.start[0],1);
	}
	sourceIndex= __shfl_sync(0xffffffff,sourceIndex,0);
	__syncwarp();

	// start loop
	S->candidate.end[0]= n_subgraph;
	// clock_t start = clock();
	while(sourceIndex < S->candidate.end[0])
	{
		int VertCount=1;
		source= S->candidate.instance_ID[sourceIndex];
		int SampleID= S->candidate.instance_ID[sourceIndex];
		int NL= G.degree_list[source];
		#ifdef profile
		// if(warpTid==0){printf("Source: %d, len: %d\n",source,NL);}
		#endif
		if((NL==0) || (NL>8000)){	
			if(warpTid==0){sourceIndex=atomicAdd(&S->candidate.start[0],1);}
			sourceIndex= __shfl_sync(0xffffffff,sourceIndex,0);
			__syncwarp();
			continue;
		}
		int len= get_neighbors(&G,source,&S->wvar[warpId],VertCount);
			
			if(NORMALIZE==0){
				select(&S->wvar[warpId],&S->cache,N,N,local_state, &G,S->count.colcount, source,S->max,bitflag,cache);
			}
			else{
				heur_normalize(&S->wvar[warpId],&S->cache,N,overN,local_state, &G,S->count.colcount, source, S->max,bitflag,cache);
				}
		frontier(&G,S,warpId,SampleID,N,source,sourceIndex, hash);
		
		__syncwarp();
		 if(warpTid==0){
			sourceIndex=atomicAdd(&S->candidate.start[0],1);
		}
		sourceIndex= __shfl_sync(0xffffffff,sourceIndex,0);
		__syncwarp();
	}
	if(tid==0){printf("%.2f,",(float)(S->max[0]/S->candidate.end[0]));}
}


int main(int args, char **argv)
{

	if(args!=11){std::cout<<"Wrong input\n"; return -1;}
	int n_blocks= atoi(argv[4]);
	int n_threads=atoi(argv[5]);
	int n_subgraph=atoi(argv[6]);
	int Normselection=atoi(argv[7]);
	int bitflag=atoi(argv[8]);
	int hash=atoi(argv[9]);
	int cache=atoi(argv[10]);
	int bias=1;
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
	
	const char *beg_file=argv[2];
	const char *csr_file=argv[3];
	const char *weight_file=argv[3];
	
	graph<long, long, long, vertex_t, index_t, weight_t>
	*ginst = new graph
	<long, long, long, vertex_t, index_t, weight_t>
	(beg_file,csr_file,weight_file);  
	gpu_graph ggraph(ginst);
	curandState *d_state;
	cudaMalloc(&d_state,sizeof(curandState));  

	Sampling *sampler;
	Sampling S(ginst->edge_count, warps, 10000,n_subgraph, BUCKETS*BUCKET_SIZE, BUCKETS);
	H_ERR(cudaMalloc((void **)&sampler, sizeof(Sampling)));

	// int *host_counter=(int *)malloc(sizeof(int));
	int *host_prefix_counter=(int *)malloc(sizeof(int));
    int *node_list=(int *)malloc(sizeof(int)*total_length);
    int *set_list=(int *)malloc(sizeof(int)*total_length);	
	
	int *degree_list=(int *)malloc(sizeof(int)*ginst->edge_count);
	std::random_device rd;
    std::mt19937 gen(57);
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
	int N=5;
	int OverN=5;
	start_time= wtime();
	check<<<n_blocks, n_threads>>>(sampler, ggraph, d_state, n_subgraph, N, OverN, Normselection,bias, bitflag, hash, cache);
	HRR(cudaDeviceSynchronize());
	total_time= wtime()-start_time;
	printf("%s,SamplingTime:%.6f\n",argv[1],total_time);


	#ifdef SelectionProfile
	if(selection==0){
		printf("Select.\n");
	for(int i=3;i<10;i+=2){
	start_time= wtime();
	check<<<n_blocks, n_threads>>>(sampler, ggraph, d_state, n_subgraph, i, 5, 0);
	HRR(cudaDeviceSynchronize());
	total_time= wtime()-start_time;
	// HRR(cudaMemcpy(host_counter,counter,sizeof(S.Wvar), cudaMemcpyDeviceToHost));
	printf("%d,%.6f\n",i,total_time);
	}}

	if(selection==1){
		printf("Normalized.\n");
	for(int i=3;i<10;i+=2){
		// for(int j=1;j<2;j++){
	start_time= wtime();
	check<<<n_blocks, n_threads>>>(sampler, ggraph, d_state, n_subgraph, i, 2, 1);
	HRR(cudaDeviceSynchronize());
	total_time= wtime()-start_time;
	printf(" %d,%d,%.6f\n",i,j,total_time);
	}}
	#endif
	// HRR(cudaMemcpy(total,g_sub_index,sizeof(int)*n_subgraph, cudaMemcpyDeviceToHost));
	// HRR(cudaMemcpy(host_prefix_counter,pre_counter,sizeof(int), cudaMemcpyDeviceToHost));
	// HRR(cudaMemcpy(host_counter,counter,sizeof(int), cudaMemcpyDeviceToHost));
	
	
	
	
	// HRR(cudaMemcpy(host_counter,counter,sizeof(int), cudaMemcpyDeviceToHost));
	// int counted=display(n_subgraph,total);
	// float rate = (float)(counted/cmp_time)/1000000;
	// printf("Kernel time:%f, Rate (Million sampled edges): %f\n",cmp_time,rate);
	// printf("%f\n",rate);
	// printf();
	//printf("Time to generate random %f\n",time_start-t1);
	//printf("Total Time:%f\n",cmp_time+ (time_start-t1));
	//r2();
	//call cuda device sync for getting error from the kernel
    //HRR(cudaMemcpy(a,d_a,sizeof(vertex_t)*5, cudaMemcpyDeviceToHost));
	//cout<<a;
	//free(total);
}  
