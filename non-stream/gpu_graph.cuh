//10/03/2016
//Graph data structure on GPUs
#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include <iostream>
#include "header.h"
#include "util.h"
#include "graph.h"

class gpu_graph
{
	public:
		vertex_t *adj_list;
		weight_t *weight_list;
		index_t *beg_pos;
		vertex_t *degree_list;
		
		index_t vert_count;
		index_t edge_count;
		index_t avg_degree;

	public:
		~gpu_graph(){}
		
		gpu_graph(
			graph<long, long, long, vertex_t, index_t, weight_t> *ginst)
		{
			vert_count=ginst->vert_count;
			edge_count=ginst->edge_count;
			avg_degree = ginst->edge_count/ginst->vert_count;

			// size_t weight_sz=sizeof(weight_t)*edge_count;
			size_t adj_sz=sizeof(vertex_t)*edge_count;
			size_t deg_sz=sizeof(vertex_t)*edge_count;
			size_t beg_sz=sizeof(index_t)*(vert_count+1);
			vertex_t *cpu_degree_list=(vertex_t*)malloc(sizeof(vertex_t)*edge_count); 
			/* Alloc GPU space */
			H_ERR(cudaMalloc((void **)&adj_list, adj_sz));
			H_ERR(cudaMalloc((void **)&degree_list, deg_sz));
			H_ERR(cudaMalloc((void **)&beg_pos, beg_sz));
			//H_ERR(cudaMalloc((void **)&weight_list, weight_sz));
			
			for(int i=0; i<(ginst->edge_count); i++)
			{
				int neighbor= ginst->adj_list[i];
				//cout<<"Index: "<<i<<"\tNeighbor: "<<neighbor<<"\n";
				cpu_degree_list[i]= ginst->beg_pos[neighbor+1] - ginst->beg_pos[neighbor];
				if((cpu_degree_list[i]>1950) & (cpu_degree_list[i]<2050))
				{
					//printf("V: %d, Degree:%d\n",neighbor,cpu_degree_list[i]);
				}
			}

			/* copy it to GPU */
			H_ERR(cudaMemcpy(adj_list,ginst->adj_list,
						adj_sz, cudaMemcpyHostToDevice));
			H_ERR(cudaMemcpy(beg_pos,ginst->beg_pos,
						beg_sz, cudaMemcpyHostToDevice));
			H_ERR(cudaMemcpy(degree_list,cpu_degree_list,
						beg_sz, cudaMemcpyHostToDevice));
			
			//H_ERR(cudaMemcpy(weight_list,ginst->weight,
			//			weight_sz, cudaMemcpyHostToDevice));
		}
};

#endif
