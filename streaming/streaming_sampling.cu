#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <set>
#include "gpu_graph.cuh"
#include "graph.h"
#include "herror.h"
#include "sampler.cuh"
#include "wtime.h"
using namespace std;

// int RAND_MAX=10000;
int sum(int length, int *a) {
  int total = 0;
  // std::cout<<"\n size:"<<length<<"\n";
  for (int i = 0; i < length; i++) {
    // std::cout<<a[i]<<"\n";
    total += a[i];
  }
  // std::cout<<"Total:"<< total <<"\n";
  return total;
}

void display(int *a, int length) {
  for (int i = 0; i < length; i++) {
    printf("%d\n", a[i]);
  }
}

__device__ void d_display(int *a, int *b, int length) {
  for (int i = 0; i < length; i++) {
    printf("Node:%d, Edge:%d\n", a[i], b[i]);
  }
}

void prefix_sum(int length, int *prefix_list) {
  for (int i = 1; i < length; i++) {
    prefix_list[i] = (prefix_list[i - 1] + prefix_list[i]);
    // printf("%dth sum, %f +  %d =
    // %f\n",i,prefix_list[i-1],arr[i],prefix_list[i]);
  }
}

__device__ int binary_search(int start, int end, float value, float *arr) {
  // printf("low:%d,high:%d,value:%f\n",start,end,value);
  int low = start;
  int high = end;
  int index = start;
  while (low <= high) {
    index = ((low + high) / 2);
    if (value < arr[index]) {
      // set high to index-1
      high = index - 1;
      // printf("high:%d\n",high);
    } else if (value > arr[index]) {
      // set low to index+1
      low = index + 1;
      // printf("low:%d\n",low);

    } else {
      break;
    }
  }
  return index;
}

__device__ int bitmap_binary_search(int start, int end, float value, float *arr,
                                    int *bitmap, int bitmap_start, int &is_in) {
  // printf("low:%d,high:%d,value:%f\n",start,end,value);
  int low = start;
  int high = end;
  int index = start;
  int bitmap_width = 32;
  while (low <= high) {
    index = ((low + high) / 2);
    if (value < arr[index]) {
      // set high to index-1
      high = index - 1;
      // printf("high:%d\n",high);
    } else if (value > arr[index]) {
      // set low to index+1
      low = index + 1;
      // printf("low:%d\n",low);
    } else {
      break;
    }
  }
  int bitmap_pos = index;
  int bit_block_index =
      bitmap_pos / bitmap_width;                  // find the address of bitmap
  int bit_block_pos = bitmap_pos % bitmap_width;  // position within a address
  // reversed------------

  // int bit_block_pos = bitmap_pos / bitmap_width;
  // int bit_block_index= bitmap_pos % bitmap_width;
  int initial_mask = 1;
  int mask = (initial_mask << bit_block_pos);
  int status = atomicOr(&bitmap[bit_block_index + bitmap_start], mask);
  is_in = (mask & status) >> bit_block_pos;

  // is_in= 0x00000001 & (status >> bit_block_pos);
  // printf("thread: %d, index:%d, bit_block_index:%d, bit_block_pos:%d,
  // mask:%d, status: %d,shift: %d,
  // is_in:%d\n",threadIdx.x,index,bit_block_index,bit_block_pos,mask,status,(mask
  // & status),is_in);
  return index;
}

void r2() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  for (int n = 0; n < 10; ++n) {
    std::cout << dis(gen) << ' ';
  }
}

__device__ float frandom(curandState *global) {
  // curand_init(1000,threadIdx.x,10,&global[threadIdx.x]);
  float x = ((curand_uniform(&global[0])));
  return x;
}

__device__ int linear_search(int neighbor, int *partition1, int *bin_count,
                             int bin, int BIN_OFFSET, int BIN_START,
                             int BUCKETS) {
  int len = bin_count[bin + BIN_OFFSET];

  int i = bin + BIN_START;
  // printf("\nL: %d, I:%d\n",len,i);
  int step = 0;
  while (step < len) {
    int test = partition1[i];
    // printf("Neighbor: %d, Test: %d, address: %d\n",neighbor,test,i);
    if (test == neighbor) {
      // printf("Duplicate detected
      // -------------------------------------------------------\n");
      return 1;
    } else {
      i += BUCKETS;
    }
    step += 1;
  }
  return 0;
}

__device__ void gpu_prefix(int total_step, int warp_tid, float *degree_l,
                           int offset_d_n, int warpsize, int len) {
  for (int i = 0; i < total_step; i++) {
    // Loop the threads
    int req_thread = len / (powf(2, (i + 1)));
    for (int iid = warp_tid; iid <= req_thread; iid += warpsize) {
      int tid_offset = iid * powf(2, i + 1);
      // calculate the index
      int i1 = (tid_offset) + (powf(2, i)) - 1 + offset_d_n;
      int i2 = (tid_offset) + powf(2, i + 1) - 1 + offset_d_n;
      if (i1 > (offset_d_n + len - 1)) {
        break;
      }
      // printf("i:%d, Index1 %d: %f,Index2 %d: %f,
      // thread:%d\n",i,i1,degree_l[i1],i2,degree_l[i2],threadIdx.x);
      // load the values to shared mem
      int temp1 = degree_l[i1];
      int temp2 = degree_l[i2];
      degree_l[i2] = temp2 + temp1;
      // printf("Index:%d, Value:%d \n",i2,temp[i2]);
    }
  }
  degree_l[len - 1 + offset_d_n] = 0;
  // printf("\nDownstep:%d\n",degree_l[len-1]);
  for (int i = (total_step - 1); i >= 0; i--) {
    // Loop the threads
    int req_thread = len / (powf(2, (i + 1)));
    for (int iid = warp_tid; iid <= req_thread; iid += warpsize) {
      int tid_offset = iid * powf(2, i + 1);
      int i1 = (tid_offset) + (powf(2, i)) - 1 + offset_d_n;
      int i2 = (tid_offset) + powf(2, i + 1) - 1 + offset_d_n;
      if (i1 > (offset_d_n + len - 1)) {
        break;
      }
      //  printf("temp1: %d, temp2: %d, thread:%d\n",i1,i2,threadIdx.x);
      // printf("Index1 %d: %f,Index2 %d: %f,
      // thread:%d\n",i1,degree_l[i1],i2,degree_l[i2],threadIdx.x);
      int temp1 = degree_l[i1];
      int temp2 = degree_l[i2];
      degree_l[i1] = temp2;
      degree_l[i2] = temp2 + temp1;
      // printf("Index:%d, Value:%d \n",i2,temp[i2]);
    }
  }
}

__global__ void check(int Graph_block_size, int streamid, int block_id,
                      vertex_t *adj_list, index_t *beg_pos,
                      weight_t *weight_list, int vertex_count,
                      curandState *global_state, int *g_node_list,
                      int *g_edge_list, int *neigh_l, float *degree_l,
                      int n_blocks, int *d_seed, int n_threads, int *total,
                      int *hashtable, int *bitmap, int total_subgraphs,
                      int *node, int *queue, int *sample_id, int *depth_tracker,
                      int *qstart_global, int *qstop_global, int *g_sub_index,
                      int n_child, int depth_limit, int sample_size, int queue_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // int __shared__ q1_start, q2_end, depth, q2_start, q2_stop;
  int temp_queue_start = qstart_global[block_id];
  int temp_queue_stop = qstop_global[block_id];
  //-----------------We may require a barrier here for storing temp
  //queue---------------------//
  int __shared__ bin_count[128];
  int warp_tid = threadIdx.x % 32;
  int G_warpID = tid / 32;
  int warpId = threadIdx.x / 32;
  int warpsize = 32;
  int offset_d_n = G_warpID * 4000;
  int BUCKETS = 32;
  int BINsize = BUCKETS * 6;
  int bitmap_size = 100;
  int Graph_block = 4;
  float prefix_time, local_d_time, global_d_time;
  clock_t start_time, stop_time;
  int __shared__ prefix;
  int seed_index;
  int BIN_OFFSET = 0;
  int depthcount, edges_traversed, q_stop, vertex, total_work;
  int q_start;
  int queue_start_address = block_id * queue_size;
  curandState local_state = global_state[threadIdx.x];
  curand_init(
      tid, 0, 0,
      &local_state);  // sequence created with different seed and same sequence
  int depth_flag = 0;
  edges_traversed = 0;
  // add all items to the combined queue: Number of threads must be greater than
  // samples
  if ((qstop_global[block_id] - qstart_global[block_id]) != 0) {
    if (warp_tid == 0) {
      q_start = atomicAdd(&qstart_global[block_id], 1);
    }
    q_start = __shfl_sync(0xffffffff, q_start, 0);
    __syncwarp();

    while (q_start < qstop_global[block_id]) {
      vertex = queue[q_start + queue_start_address];
      //if(warp_tid==0){printf("Block_id:%d, StreamId: %d, G_warpID: %d,SampleID:%d, vertex:%d, q_stop:%d,q_start:%d,depth:%d\n",block_id,streamid,G_warpID,sample_id[q_start+queue_start_address],vertex,qstop_global[block_id],q_start,depth_tracker[q_start+queue_start_address]);}
      int neighbor_start = beg_pos[vertex];
      int neighbor_end = beg_pos[vertex + 1];
      int neighbor_length = neighbor_end - neighbor_start;
      edges_traversed += neighbor_length;
      if (neighbor_length == 0) {
        if (warp_tid == 0) {
          q_start = atomicAdd(&qstart_global[block_id], 1);
        }
        q_start = __shfl_sync(0xffffffff, q_start, 0);
        __syncwarp();
        continue;
      }
      int is_in = 0;
      int new_neighbor;
      int selected = 0;
      if (neighbor_length < n_child) {
        prefix = 0;
      } else {
        prefix = 1;
      }
      int thread_flag = 0;
      if ((warp_tid < n_child) && (warp_tid < neighbor_length)) {
        thread_flag = 1;
      }
      if (prefix) {
        // For each neighbor, calculate the degree of its neighbor
        int index = offset_d_n + warp_tid;  // use block and thread Id for index
        for (int i = warp_tid + neighbor_start; i < neighbor_end;
             i += warpsize) {
          // neighbor ID
          int temp = adj_list[i];
          // if((temp>Graph_block_size)& (warp_tid==0)){printf("Reading from
          // outside.\n");} degree of neighbor
          degree_l[index] = float(beg_pos[temp + 1] - beg_pos[temp]);
          // printf("%d has a degree of %f found by
          // %d,index:%d\n",temp,degree_l[index],threadIdx.x,index);
          index += warpsize;
        }
        int i_start_neigh = offset_d_n;
        int i_end_neigh = i_start_neigh + neighbor_length;
        //	printf("Starting prefix_sum\n");
        // start_time = clock();
        float bits = log2f(neighbor_length);
        int raise = ceilf(bits);
        int max_bit = powf(2, raise);
        int len = max_bit;
        int total_step = log2f(max_bit);
        gpu_prefix(total_step, warp_tid, degree_l, offset_d_n, warpsize, len);
        float sum = degree_l[neighbor_length - 1 + offset_d_n];
        for (int i = warp_tid + i_start_neigh; i < i_end_neigh; i += warpsize) {
          // printf("i:%d, degree:%.2f\n",i,degree_l[i]);
          degree_l[i] = degree_l[i] / ((double)sum);
        }
        // start_time = clock();
        int bitmap_start = G_warpID * bitmap_size;
        if (warp_tid < n_child) {
          float r = curand_uniform(&local_state);
          //------------------------------------Using
          //bitmaps----------------------------------------------
          selected =
              bitmap_binary_search(i_start_neigh, i_end_neigh, r, degree_l,
                                   bitmap, bitmap_start, is_in);
          new_neighbor = adj_list[selected + neighbor_start - offset_d_n];
          // if(is_in==0) {printf("Index: %d, New N: %d, Thread:
          // %d\n",selected,new_neighbor,threadIdx.x);}
          //--------------------------------------------------------------------------------------------
        }
        // Reset Bitmaps
        int start = bitmap_start + warp_tid;
        int end = bitmap_start + bitmap_size;
        for (int i = start; i < end; i += warpsize) {
          bitmap[i] = 0;
          // printf("Bitmap cleared at %d\n",i);
        }
      }
      else {
        if (thread_flag) {
          new_neighbor =
              adj_list[warp_tid + neighbor_start];  // unwanted thread also may
                                                    // get some child but will be
                                                    // neglected in next section
        }
        // printf("New Neighbor: %d, thread: %d\n",new_neighbor,threadIdx.x);
      }
      /* Use hashtable for detecting duplicates*/
      int BIN_START = sample_id[q_start] * BINsize;
      if (is_in == 0 && thread_flag) {
        int bin = new_neighbor % BUCKETS;
        is_in = linear_search(new_neighbor, hashtable, bin_count, bin,
                              BIN_OFFSET, BIN_START, BUCKETS);
        // if(is_in==1){printf("Duplicated Found: %d\n",new_neighbor);}
      }
      //-------------------------------------------------------------------
      if (is_in == 0 && thread_flag) {
        //------------------------Store in
        //hashtable-----------------------------//
        int bin = new_neighbor % BUCKETS;
        // int index= warpId;
        int index = atomicAdd(&bin_count[bin + BIN_OFFSET], 1);
        hashtable[index] = new_neighbor;
        hashtable[index * BUCKETS + bin + BIN_START] = new_neighbor;
        int g_sub_start = sample_id[q_start] * sample_size;
        int g_to = atomicAdd(&g_sub_index[sample_id[q_start]], 1);
        //g_node_list[g_to + g_sub_start] = vertex;
        //g_edge_list[g_to + g_sub_start] = new_neighbor;
        printf("%d,%d,%d,%d\n",vertex,new_neighbor,sample_id[q_start],depth_tracker[q_start + queue_start_address]);
	//Added to sample:752601,328138,20,0,2
	// add to the expand queue
        if (depth_tracker[q_start] < depth_limit) {
          int new_bin = new_neighbor / Graph_block_size;
          int new_queue_start = new_bin * queue_size;
	  // if(new_bin!=0)
          // { printf("Block:%d, Added to block:%d\n",block_id,new_bin);}
          int to = atomicAdd(&qstop_global[new_bin], 1);
          queue[to + new_queue_start] = new_neighbor;
          sample_id[to + new_queue_start] =
              sample_id[q_start + queue_start_address];
          depth_tracker[to + new_queue_start] =
              depth_tracker[q_start + queue_start_address] + 1;
         //printf("Added: %d,  to queue at index %d and block %d, local_index: %d, offset: %d, new_d: %d, prev_d: %d\n",new_neighbor,to + new_queue_start,new_bin, to, new_queue_start,depth_tracker[to + new_queue_start], depth_tracker[q_start + queue_start_address]);
	 }
      }
      // q_start+=1;
      if ((qstart_global[block_id] > qstop_global[block_id])) {
        break;
      }
      if (warp_tid == 0) {
        q_start = atomicAdd(&qstart_global[block_id], 1);
      }
      q_start = __shfl_sync(0xffffffff, q_start, 0);
      __syncwarp();
    }
  }
}

int build_histogram(int n_subgraph, int *input, int *frequency,
                    int block_window_size, int block_size, int vert_count,
                    int vertex_block_count) {
  int max_index = 0, max_value = 0;
  for (int i = 0; i < n_subgraph; i++) {
    int block = input[i] / block_size;
    if (block > vertex_block_count) {
      block = vertex_block_count;
    }
    // cout<<"Value:"<<input[i]<<"\tBlock:"<<block<<"\n";
    frequency[block] += 1;
  }
  // display(frequency,vertex_block_count);
  prefix_sum(vertex_block_count, frequency);
  // display(frequency,vertex_block_count);
  for (int j = 0; j < (vertex_block_count - 5); j += block_window_size) {
    int combined_freq = frequency[j + block_window_size - 1] - frequency[j];
    if (combined_freq > max_index) {
      max_index = j;
      max_value = combined_freq;
    }
  }
  cout << "Max_index:" << max_index << "Max_value:" << max_value << "\n";
  return max_index;
}

int block_augument(int blocks, int vertex_count, index_t *beg_pos,
                   int *beg_size_list, int *adj_size_list) {
  int block_size = (vertex_count) / blocks;
  for (int i = 0; i < (blocks + 1); i += 1) {
    int start_block = i * block_size;
    if (i == blocks) {
      start_block = vertex_count;
    }
    beg_size_list[i] = start_block;
    int start_adj = beg_pos[block_size * i];
    adj_size_list[i] = start_adj;
  }
  return 0;
}

struct arguments Sampler(char beg[100], char csr[100], int n_blocks,
                         int n_threads, int n_subgraph, int frontier_size,
                         int neighbor_size, int depth, struct arguments args,
                         int rank) {
  // if(args!=7){std::cout<<"Wrong input\n"; return -1;}
  //n_child, depth, each_subgraph, queue_size
  // cout<<"\nblocks:"<<n_blocks<<"\tThreads:"<<n_threads<<"\tSubgraphs:"<<n_subgraph<<"\n";
  // int n_threads=32;
  int *total = (int *)malloc(sizeof(int) * n_subgraph);
  int len = 5;
  int T_Group = n_threads / 32;
  int n_child = neighbor_size;
  int each_subgraph = depth * n_child;
  int total_length = each_subgraph * n_subgraph;
  int neighbor_length_max = n_blocks * 6000 * T_Group;
  int PER_BLOCK_WARP = T_Group;
  int BUCKET_SIZE = 10;
  int BUCKETS = 128;
  int total_mem_for_hash = n_blocks * PER_BLOCK_WARP * BUCKETS * BUCKET_SIZE;
  int total_mem_for_bitmap = n_blocks * PER_BLOCK_WARP * 300;
  int queue_size = 10000;
  int Graph_block = 4;
  int total_queue_memory = queue_size * Graph_block * n_subgraph;

  // std::cout<<"Input: ./exe beg csr nblocks nthreads\n";
  const char *beg_file = beg;
  const char *csr_file = csr;
  const char *weight_file = csr;  // unnecessary
  // template <file_vertex_t, file_index_t, file_weight_t
  // new_vertex_t, new_index_t, new_weight_t>
  graph<long, long, long, vertex_t, index_t, weight_t> *ginst =
      new graph<long, long, long, vertex_t, index_t, weight_t>(
          beg_file, csr_file, weight_file);
  int vertex_count = ginst->vert_count;
  int edge_count = ginst->edge_count;
  int Graph_block_size = vertex_count / Graph_block;
  // int Graph_block_size=2000;
  /*
  printf("Size of blocks\n");
  for (int i = 0; i < 4; i++) {
    printf("%d,%d\n", i,
           ginst->beg_pos[(i + 1) * Graph_block_size] -
               ginst->beg_pos[(i)*Graph_block_size]);
  }
  */
  curandState *d_state;
  cudaMalloc(&d_state, sizeof(curandState));
  gpu_graph ggraph(ginst);
  int *node_list = (int *)malloc(sizeof(int) * total_length);
  int *set_list = (int *)malloc(sizeof(int) * total_length);
  float *n_random = (float *)malloc(sizeof(float) * n_threads);
  int *seeds = (int *)malloc(sizeof(int) * total_queue_memory);
  int *seeds_counter = (int *)malloc(sizeof(int) * Graph_block);
  int *start_queue = (int *)malloc(sizeof(int) * Graph_block);
  int *degree_list = (int *)malloc(sizeof(int) * ginst->edge_count);
  int *adj_size_list = (int *)malloc(sizeof(int) * (Graph_block + 1));
  int *beg_size_list = (int *)malloc(sizeof(int) * (Graph_block + 1));
  for (int n = 0; n < Graph_block; n++) {
    seeds_counter[n] = 0;
    start_queue[n] = 0;
  }
  std::random_device rd;
  // 200 --> 370 Mteps
  int numBlocks;
  // cudaGetDevice(&device);
  // cudaGetDeviceProperties(&prop, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, check, n_threads,
                                                0);

  int deviceCount;
  HRR(cudaGetDeviceCount(&deviceCount));
  // printf("My rank: %d, totaldevice: %d\n", rank,deviceCount);
  // HRR(cudaSetDevice(rank%deviceCount));
  // cout<<"Max allocatable Blocks:"<<numBlocks<<"\n";
  int *d_node_list;
  int *d_edge_list;
  int *d_neigh_l;
  float *d_degree_l;
  // float *d_random;
  int *d_seed;
  int *d_total;
  for (int i = 0; i < (ginst->edge_count); i++) {
    int neighbor = ginst->adj_list[i];
    degree_list[i] = ginst->beg_pos[neighbor + 1] - ginst->beg_pos[neighbor];
  }
  int *hashtable, *bitmap, *node, *queue, *qstop_global, *qstart_global,
      *sample_id, *depth_tracker, *g_sub_index, *degree_l, *prefix_status;
  // Size of blocks
  HRR(cudaMalloc((void **)&d_total, sizeof(int) * n_subgraph));
  HRR(cudaMalloc((void **)&node, sizeof(int) * 2));
  HRR(cudaMalloc((void **)&degree_l, sizeof(int) * ginst->edge_count));
  HRR(cudaMalloc((void **)&prefix_status, sizeof(int) * ginst->edge_count));
  HRR(cudaMalloc((void **)&d_degree_l, sizeof(float) * ginst->edge_count));
  HRR(cudaMalloc((void **)&qstart_global, sizeof(int) * Graph_block));
  HRR(cudaMalloc((void **)&qstop_global, sizeof(int) * Graph_block));
  HRR(cudaMalloc((void **)&d_node_list, sizeof(int) * total_length));
  HRR(cudaMalloc((void **)&d_edge_list, sizeof(int) * total_length));
  HRR(cudaMalloc((void **)&d_neigh_l, sizeof(int) * neighbor_length_max));
  HRR(cudaMalloc((void **)&hashtable, sizeof(int) * total_mem_for_hash));
  HRR(cudaMalloc((void **)&bitmap, sizeof(int) * total_mem_for_bitmap));
  HRR(cudaMalloc((void **)&d_degree_l, sizeof(float) * neighbor_length_max));
  HRR(cudaMalloc((void **)&queue, sizeof(int) * total_queue_memory));
  HRR(cudaMalloc((void **)&sample_id, sizeof(int) * total_queue_memory));
  HRR(cudaMalloc((void **)&depth_tracker, sizeof(int) * total_queue_memory));
  HRR(cudaMalloc((void **)&g_sub_index, sizeof(float) * total_queue_memory));
  int *h_sample_id = (int *)malloc(sizeof(int) * total_queue_memory);
  int *h_depth_tracker = (int *)malloc(sizeof(int) * total_queue_memory);
  std::mt19937 gen(57);
  std::uniform_int_distribution<> dis(1, vertex_count / 4);

  for (int n = 0; n < n_subgraph; n++) {
    int new_seed = dis(gen);
    int bin_new = new_seed / Graph_block_size;
    if (bin_new > Graph_block) {
      bin_new = Graph_block;
    }
    int pos = bin_new * (queue_size) + seeds_counter[bin_new];
    assert(pos < total_queue_memory);
    seeds_counter[bin_new]++;
    seeds[pos] = new_seed;
    h_sample_id[pos] = n;
    h_depth_tracker[pos] = 0;
    // printf("N_subgraph: %d, Seed:%d, Bin:%d\n",n,new_seed,bin_new);
  }
  /*	For streaming partition */

  HRR(cudaMemcpy(queue, seeds, sizeof(int) * total_queue_memory,
                 cudaMemcpyHostToDevice));
  HRR(cudaMemcpy(qstart_global, start_queue, sizeof(int) * Graph_block,
                 cudaMemcpyHostToDevice));
  HRR(cudaMemcpy(qstop_global, seeds_counter, sizeof(int) * Graph_block,
                 cudaMemcpyHostToDevice));
  HRR(cudaMemcpy(sample_id, h_sample_id, sizeof(int) * total_queue_memory,
                 cudaMemcpyHostToDevice));
  HRR(cudaMemcpy(depth_tracker, h_depth_tracker,
                 sizeof(int) * total_queue_memory, cudaMemcpyHostToDevice));
  // create three cuda streams

  cudaStream_t stream1, stream2, stream3, stream4;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);
  cudaEvent_t event;
  cudaEventCreate(&event);
  // find top 3 blocks
  int sampling_complete = false;
  int i = 0, block_id1 = 0, block_id2 = 1, block_id3 = 2, block_id4 = 3;

  int q_count, max, value;
  block_augument(Graph_block, vertex_count, ginst->beg_pos, beg_size_list,
                 adj_size_list);
  int *block_active = (int *)malloc(sizeof(int) * (Graph_block));
  int *frontiers_count = (int *)malloc(sizeof(int) * (Graph_block));

  for (int j = 0; j < Graph_block; j++) {
    frontiers_count[j] = seeds_counter[j] - start_queue[j];
    // printf("Value: %d, j: %d,Q_count:\n",frontiers_count[j],j);
    if (frontiers_count[j] == 0) {
      block_active[j] = 0;
    } else {
      block_active[j] = 1;
    }
  }
  printf("<Sampled edges in csv format.>\nsource, destination, sample_id, depth\n");
  // display(block_active,Graph_block);
  // block[1]=1;
  // block[2]=1;
  // printf("Start while loop.\n");
  double time_start = wtime();
  while (sampling_complete == false) {
    // display(block_active,Graph_block);
    if (1) {
      H_ERR(cudaMemcpyAsync(&ggraph.adj_list[adj_size_list[block_id1]],
                            &ginst->adj_list[adj_size_list[block_id1]],
                            adj_size_list[block_id2] - adj_size_list[block_id1],
                            cudaMemcpyHostToDevice, stream1));
      H_ERR(cudaMemcpyAsync(&ggraph.beg_pos[beg_size_list[block_id1]],
                            &ginst->beg_pos[beg_size_list[block_id1]],
                            beg_size_list[block_id2] - beg_size_list[block_id1],
                            cudaMemcpyHostToDevice, stream1));

      check<<<n_blocks, n_threads, 0, stream1>>>(
          Graph_block_size, 0, block_id1, ggraph.adj_list, ggraph.beg_pos,
          ggraph.weight_list, ggraph.vert_count, d_state, d_node_list,
          d_edge_list, d_neigh_l, d_degree_l, n_blocks, d_seed, n_threads,
          d_total, hashtable, bitmap, n_subgraph, node, queue, sample_id,
          depth_tracker, qstart_global, qstop_global, g_sub_index, 
	  n_child, depth, each_subgraph, queue_size);
    }

    if (block_active[1]) {
      H_ERR(cudaMemcpyAsync(&ggraph.adj_list[adj_size_list[block_id2]],
                            &ginst->adj_list[adj_size_list[block_id2]],
                            adj_size_list[block_id3] - adj_size_list[block_id2],
                            cudaMemcpyHostToDevice, stream2));
      H_ERR(cudaMemcpyAsync(&ggraph.beg_pos[beg_size_list[block_id2]],
                            &ginst->beg_pos[beg_size_list[block_id2]],
                            beg_size_list[block_id3] - beg_size_list[block_id2],
                            cudaMemcpyHostToDevice, stream2));
      check<<<n_blocks, n_threads, 1, stream2>>>(
          Graph_block_size, 1, block_id2, ggraph.adj_list, ggraph.beg_pos,
          ggraph.weight_list, ggraph.vert_count, d_state, d_node_list,
          d_edge_list, d_neigh_l, d_degree_l, n_blocks, d_seed, n_threads,
          d_total, hashtable, bitmap, n_subgraph, node, queue, sample_id,
          depth_tracker, qstart_global, qstop_global, g_sub_index,
	  n_child, depth, each_subgraph, queue_size);
    }

    if (block_active[2]) {
      H_ERR(cudaMemcpyAsync(&ggraph.adj_list[adj_size_list[block_id3]],
                            &ginst->adj_list[adj_size_list[block_id3]],
                            adj_size_list[block_id4] - adj_size_list[block_id3],
                            cudaMemcpyHostToDevice, stream3));
      H_ERR(cudaMemcpyAsync(&ggraph.beg_pos[beg_size_list[block_id3]],
                            &ginst->beg_pos[beg_size_list[block_id3]],
                            beg_size_list[block_id4] - beg_size_list[block_id3],
                            cudaMemcpyHostToDevice, stream3));
      check<<<n_blocks, n_threads, 2, stream3>>>(
          Graph_block_size, 2, block_id3, ggraph.adj_list, ggraph.beg_pos,
          ggraph.weight_list, ggraph.vert_count, d_state, d_node_list,
          d_edge_list, d_neigh_l, d_degree_l, n_blocks, d_seed, n_threads,
          d_total, hashtable, bitmap, n_subgraph, node, queue, sample_id,
          depth_tracker, qstart_global, qstop_global, g_sub_index,
	  n_child, depth, each_subgraph, queue_size);
    }

    if (block_active[3]) {
      H_ERR(cudaMemcpyAsync(&ggraph.adj_list[adj_size_list[block_id4]],
                            &ginst->adj_list[adj_size_list[block_id4]],
                            adj_size_list[4] - adj_size_list[block_id4],
                            cudaMemcpyHostToDevice, stream4));
      H_ERR(cudaMemcpyAsync(&ggraph.beg_pos[beg_size_list[block_id4]],
                            &ginst->beg_pos[beg_size_list[block_id4]],
                            beg_size_list[4] - beg_size_list[block_id4],
                            cudaMemcpyHostToDevice, stream4));
      check<<<n_blocks, n_threads, 3, stream4>>>(
          Graph_block_size, 3, block_id4, ggraph.adj_list, ggraph.beg_pos,
          ggraph.weight_list, ggraph.vert_count, d_state, d_node_list,
          d_edge_list, d_neigh_l, d_degree_l, n_blocks, d_seed, n_threads,
          d_total, hashtable, bitmap, n_subgraph, node, queue, sample_id,
          depth_tracker, qstart_global, qstop_global, g_sub_index,
	  n_child, depth, each_subgraph, queue_size);
    }
    // wait for completion
    // find new top 3 blocks
    int status1 = cudaStreamQuery(stream1);
    // cout<<"Status1: "<<status1<<"\n";
    int status2 = cudaStreamQuery(stream2);
    // cout<<"Status2: "<<status2<<"\n";
    HRR(cudaDeviceSynchronize());
    HRR(cudaMemcpy(start_queue, qstart_global, sizeof(int) * Graph_block,
                   cudaMemcpyDeviceToHost));
    HRR(cudaMemcpy(seeds_counter, qstop_global, sizeof(int) * Graph_block,
                   cudaMemcpyDeviceToHost));
    max = 0, value = 0;
    q_count = 0;
    // display(seeds_counter,Graph_block);
    // display(start_queue,Graph_block);
    int max_value = 0;
    for (int j = 0; j < Graph_block; j++) {
      frontiers_count[j] = seeds_counter[j] - start_queue[j];
      if (frontiers_count[j] <= 0) {
        block_active[j] = 0;
        frontiers_count[j] = 0;
      }
      q_count += frontiers_count[j];
      if (max_value < frontiers_count[j]) {
        max = j;
        max_value = frontiers_count[j];
      }
      //printf("Value: %d, j: %d,Q_count: %d, max: %d\n", frontiers_count[j], j,
      //       q_count, max);
    }
    i++;
    block_active[max] = 1;
    //printf("Value of i:%d\n", i);

    if (q_count == 0) {  // printf("Sampling complete;\n");
      sampling_complete = true;
    }
    if (i == 10) {
      // printf("Sampling complete;\n");
      sampling_complete = true;
    }
    // printf("Total: %d,max:%d, value:%d, Value of i:
    // %d\n",q_count,max,value,i);
  }
  HRR(cudaDeviceSynchronize());
  double cmp_time = wtime() - time_start;
  HRR(cudaMemcpy(total, g_sub_index, sizeof(int) * n_subgraph,
                 cudaMemcpyDeviceToHost));
  int counted = sum(n_subgraph, total);
  float rate = (float)(counted / cmp_time) / 1000000;
  // printf("%s,Kernel time:%f, Rate (Million sampled edges):
  // %f\n",argv[1],cmp_time,rate); printf("%s,Samples: %d,
  // time:%f\n",argv[1],n_subgraph,cmp_time);
  printf("<End of edge list>\n");
  args.sampled_edges = counted;
  args.time = cmp_time;
  return args;
}

// void blocks_allocator(int n_blocks,int *Block,  )
