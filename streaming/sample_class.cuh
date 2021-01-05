#ifndef SAMPLER_H
#define SAMPLER_H

#include "herror.h"
#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

class Cd{
    /*
        Candidate list shared by all instances
    */
public:
    int s=5;
    int *instance_ID, *vertices, *depth;
    int *start, *end;
    ~Cd(){};
    Cd(){};
    Cd(int len ){
        H_ERR(cudaMalloc((void **)&instance_ID, sizeof(int)*len));
        H_ERR(cudaMalloc((void **)&vertices, sizeof(int)*len));
        H_ERR(cudaMalloc((void **)&depth, sizeof(int)*len));
        H_ERR(cudaMalloc((void **)&start, sizeof(int)*2));
        H_ERR(cudaMalloc((void **)&end, sizeof(int)*2));
    }
};
 
class Dimnesion{
public:
    int *pool;
    ~Dimnesion(){};
    Dimnesion(){};
    void init(int FrontierSize){
        H_ERR(cudaMalloc((void **)&pool, sizeof(int)*FrontierSize));
    }
};

class Wv{
    /*
        Warp variables 
    */
public:
    int test=1;
    int *total_counter;
    int *frontier, *findex;
    int *neighbors, *nindex;
    float *degree;
    int *dindex;
    int *selected, *sindex;
    int *tempSelected;
    int *bitmap, *bindex;
    int NL, NS;
    int *max;
    ~Wv(){};
    Wv(){}
    void init(int flen,int nlen, int dlen, int slen){
        H_ERR(cudaMalloc((void **)&frontier, sizeof(int)*flen));
        H_ERR(cudaMalloc((void **)&neighbors, sizeof(int)*nlen));
        H_ERR(cudaMalloc((void **)&degree, sizeof(float)*dlen));
        H_ERR(cudaMalloc((void **)&bitmap, sizeof(int)*(dlen/32)));
        H_ERR(cudaMalloc((void **)&selected, sizeof(int)*slen));
        H_ERR(cudaMalloc((void **)&selected, sizeof(int)*slen));
        H_ERR(cudaMalloc((void **)&findex, sizeof(int)*2));
        H_ERR(cudaMalloc((void **)&nindex, sizeof(int)*2));
        H_ERR(cudaMalloc((void **)&dindex, sizeof(int)*2));
        H_ERR(cudaMalloc((void **)&sindex, sizeof(int)*2));
        H_ERR(cudaMalloc((void **)&bindex, sizeof(int)*2));
        H_ERR(cudaMalloc((void **)&max, sizeof(int)*2));
        H_ERR(cudaMalloc((void **)&total_counter, sizeof(int)*2));
    }
};



class Si{
    /*
        sampled graph for instances. Each instance have its own sample graph.
    */
public:
    int *vertex,*edge;
    int *start;
    ~Si(){};
    Si(){}
    void init(int len){
        H_ERR(cudaMalloc((void **)&vertex, sizeof(int)*len));
        H_ERR(cudaMalloc((void **)&edge, sizeof(int)*len));
        H_ERR(cudaMalloc((void **)&start, sizeof(int)*2));
    }
};
 
class Ht{
    /*
        Hashtable for each instance
    */
public:
    int *hash;
    int *bin_counter;
    int BUCKETS;
    int bin_size=125;
    ~Ht(){};
    Ht(){}
    void init(int bin_count){
        BUCKETS=bin_count;
        H_ERR(cudaMalloc((void **)&hash, sizeof(int)*bin_count*bin_size));
        H_ERR(cudaMalloc((void **)&bin_counter, sizeof(int)*bin_count));
    }
};

class Co{
    /*
        Counters used in sampling.
    */
public:
    int *counter, *pre_counter, *total, *colcount, *max;
    ~Co(){};
    Co(){};
    Co(int total){
        HRR(cudaMalloc((void **) &counter,sizeof(int)*2));
        HRR(cudaMalloc((void **) &max,sizeof(int)*8000));
        HRR(cudaMalloc((void **) &pre_counter,sizeof(int)*2));
        HRR(cudaMalloc((void **) &colcount,sizeof(int)*50));
        HRR(cudaMalloc((void **) &total,sizeof(int)*total));
    }
};

class Cp{
    /*
        Cache probability for each vertex in the graph.
    */
public:
    int *status;
    float *probability;
    int *counter;
    ~Cp(){};
    Cp(){};
    Cp(int len){
        // HRR(cudaMalloc((void **) &status,sizeof(int)*len));
        // HRR(cudaMalloc((void **) &probability,sizeof(float)*len));
        HRR(cudaMalloc((void **) &counter,sizeof(int)*2));
    }
};


class Sampling{
    /*
        Collection of objects for sampling 
    */
public:
    Cd candidate;
    Si samples[20100];
    Ht hashtable[20100];
    Co count;
    Wv wvar[2000];
    Dimnesion front[4000];
    Cp cache;
    int *max,*sampled_count,*frontier_degree;
    int n_child=1;
	int DEPTH_LIMIT;
	int BUCKETS=32;
    ~Sampling(){};
    Sampling(int edgecount,int warpCount, int qlen, int seeds, int C_len, int sampleSize, int FrontierSize, int depth){
        DEPTH_LIMIT=depth;
        count=  Co(seeds);
        candidate= Cd(seeds*8000);
        cache= Cp(edgecount);
        HRR(cudaMalloc((void **) &max,sizeof(int)*2));
        HRR(cudaMalloc((void **) &frontier_degree,sizeof(int)*sampleSize*FrontierSize));
        HRR(cudaMalloc((void **) &sampled_count,sizeof(int)));
        for(int i=0;i<seeds;i++)
        {
            samples[i].init(sampleSize);
            hashtable[i].init(BUCKETS);   
            front[i].init(FrontierSize);
        }

        for(int i=0;i<warpCount;i++)
        {
            // queue[i].init(qLen);
            wvar[i].init(seeds,21000,21000,BUCKETS);
        }
    }
};

class Layer_sampling{
    /*
        Collection of objects for sampling 
    */
public:
    Cd candidate;
    Si samples[20100];
    Co count;
    Wv wvar[2000];
    int *max,*sampled_count,*frontier_degree;
    int n_child=1;
    int DEPTH_LIMIT;
    int BUCKETS=32;
    ~Layer_sampling(){};
    Layer_sampling(int edgecount,int warpCount, int qlen, int seeds, int C_len, int sampleSize, int FrontierSize, int depth){
        DEPTH_LIMIT=depth;
        count=  Co(seeds);
        candidate= Cd(seeds*8000);
        HRR(cudaMalloc((void **) &max,sizeof(int)*2));
        HRR(cudaMalloc((void **) &frontier_degree,sizeof(int)*sampleSize*FrontierSize));
        HRR(cudaMalloc((void **) &sampled_count,sizeof(int)));
        for(int i=0;i<seeds;i++)
        {
            samples[i].init(sampleSize*FrontierSize);
            // front[i].init(FrontierSize);
        }

        for(int i=0;i<warpCount;i++)
        {
            // queue[i].init(qLen);
            wvar[i].init(seeds,21000,21000,BUCKETS);
        }
    }
};


#endif