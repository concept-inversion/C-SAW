#ifndef FNC
#define FNC
#include "herror.h"
#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "sampler.cuh"
#include "api.cuh"
#define profil
__device__
int binary_search(int start,int end,float value, float *arr)
{
    //printf("low:%d,high:%d,value:%f\n",start,end,value); 
    int low=start;
    int high=end;
    int index=start;
	
	// atomicAdd(&counter[0],1);
	while (low<=high)
    {
		index=((low+high)/2);
        if (value<arr[index])
		{
            //set high to index-1
            high= index-1;
	    //printf("high:%d\n",high);
        }
        else if (value>arr[index])
        {
            // set low to index+1
            low = index+1;
            //printf("low:%d\n",low);

		}
        else
        {
            break;
        } 
        
    }
    return index;
}    

__device__
int bitmap_search(int *bitmap, int bitmap_start, int index)
{
    int bitmap_width=32;
    int bitmap_pos= index;
	// #ifdef not_reversed
	int bit_block_index = bitmap_pos / bitmap_width;   // find the address of bitmap
	int bit_block_pos= bitmap_pos % bitmap_width;		// position within a address
	// #endif
	// reversed------------

	//#ifdef reversed
	// int bit_block_pos = bitmap_pos / bitmap_width;
	// int bit_block_index= bitmap_pos % bitmap_width;
	//#endif

	int initial_mask=1;
	int mask = (initial_mask << bit_block_pos);
	int status=atomicOr(&bitmap[bit_block_index+bitmap_start],mask);
	// int status=mask;
	int is_in= (mask & status) >> bit_block_pos;
	if(is_in!=0){is_in=1;}
	//is_in= 0x00000001 & (status >> bit_block_pos);
	//printf("thread: %d, index:%d, bit_block_index:%d, bit_block_pos:%d, mask:%d, status: %d,shift: %d, is_in:%d\n",threadIdx.x,index,bit_block_index,bit_block_pos,mask,status,(mask & status),is_in);
    return is_in;
}

__device__
int linear_search(int *bitmap, int bitmap_start, int index)
{

    int warpTID=threadIdx.x%32;
    int pos= warpTID;
    int temp_status= 0;
    while(pos<256){
        if (bitmap[index]==1)
        {
            temp_status=1;
        }
        pos+=warpSize;
    }

    int bitmap_width=32;
    int bitmap_pos= index;
	// #ifdef not_reversed
	int bit_block_index = bitmap_pos / bitmap_width;   // find the address of bitmap
	int bit_block_pos= bitmap_pos % bitmap_width;		// position within a address
	// #endif
	// reversed------------

	//#ifdef reversed
	// int bit_block_pos = bitmap_pos / bitmap_width;
	// int bit_block_index= bitmap_pos % bitmap_width;
	//#endif

	int initial_mask=1;
	int mask = (initial_mask << bit_block_pos);
	int status=atomicOr(&bitmap[bit_block_index+bitmap_start],mask);
	// int status=mask;
	int is_in= (mask & status) >> bit_block_pos;
	if(is_in!=0){is_in=1;}
	//is_in= 0x00000001 & (status >> bit_block_pos);
	//printf("thread: %d, index:%d, bit_block_index:%d, bit_block_pos:%d, mask:%d, status: %d,shift: %d, is_in:%d\n",threadIdx.x,index,bit_block_index,bit_block_pos,mask,status,(mask & status),is_in);
    return is_in;
}


__device__
void gpu_prefix(int total_step,int warp_tid,float *degree_l, int offset_d_n, int warpsize, int len)
{
    warpsize=32;
	for (int i=0; i< total_step; i++)
	{
		// Loop the threads
		int req_thread = len/(powf(2,(i+1)));
		for (int iid= warp_tid; iid<=req_thread; iid+=warpsize)
		{
		
			int tid_offset = iid*powf(2,i+1);
			// calculate the index
			int i1= (tid_offset) +(powf(2,i))-1+offset_d_n;
			int i2= (tid_offset) +powf(2,i+1)-1+offset_d_n;
			if(i1> (offset_d_n+len-1)){break; }
			//printf("i:%d, Index1 %d: %f,Index2 %d: %f, thread:%d\n",i,i1,degree_l[i1],i2,degree_l[i2],threadIdx.x);
			// load the values to shared mem
			int temp1= degree_l[i1];
			int temp2= degree_l[i2];
			degree_l[i2] = temp2+ temp1;
			//printf("Index:%d, Value:%d \n",i2,temp[i2]);
		}
    }
    // __syncthreads();
	degree_l[len-1+offset_d_n]=0;
	//printf("\nDownstep:%d\n",degree_l[len-1]);
	for (int i=(total_step-1);i >= 0; i--  )
	{
		// Loop the threads
		int req_thread = len/(powf(2,(i+1)));
		for (int iid= warp_tid; iid<=req_thread; iid+=warpsize)
		{
			int tid_offset = iid * powf(2,i+1);
			int i1= (tid_offset) + (powf(2,i))-1+offset_d_n;
			int i2= (tid_offset) + powf(2,i+1)-1+offset_d_n;
			if(i1 > (offset_d_n+len-1)){break;}
			//  printf("temp1: %d, temp2: %d, thread:%d\n",i1,i2,threadIdx.x);
			// printf("Index1 %d: %f,Index2 %d: %f, thread:%d\n",i1,degree_l[i1],i2,degree_l[i2],threadIdx.x);
			int temp1 = degree_l[i1];
			int temp2 = degree_l[i2];
			degree_l[i1]=temp2;
			degree_l[i2]=temp2+temp1;
			//printf("Index:%d, Value:%d \n",i2,temp[i2]);
		}	
	}
}

__device__ void
ITS(float *degree_l,int offset_d_n,int warpsize, int neighbor_length){
    float bits = log2f(neighbor_length);
	int raise = ceilf(bits);
	int max_bit = powf(2,raise);
	int len=max_bit;
    int total_step= log2f(max_bit);
    int warp_tid = threadIdx.x%32;
    // __syncthreads();
    gpu_prefix(total_step,warp_tid,degree_l,offset_d_n,warpsize,len);
    float sum = degree_l[neighbor_length-1+offset_d_n];
    for (int i = warp_tid; i < neighbor_length; i+=warpsize)
    {	
        degree_l[i]=degree_l[i]/((double)sum);
        // printf("i:%d, degree:%.2f\n",i,degree_l[i]);
    }
}

__device__ int
max(int *data, int len)
{
	int max=data[0];
	for(int i=0;i<len;i++)
	{
        if (data[i]>max){max=data[i];}
        // printf("data: %d\n",data[i]);
	}
	return max;
}

__device__ void
read_prob(float *degree_l, float *prob, int len, int offset){
    int index=threadIdx.x %32;
    while(index<len)
    {
        degree_l[index]= prob[index + offset];
        index+=warpSize;
    }
}

__device__ void
write_prob(float *degree_l, float *prob, int len, int offset){
    int index=threadIdx.x %32;
    while(index<len)
    {
        prob[index + offset]= degree_l[index];
        index+=warpSize;
    }
}

__device__ int
rand_integer(curandState local_state, int MAX)
{
    int num= curand(&local_state)%MAX;
    // printf("Tid: %d, Rand num: %d\n",threadIdx.x, num);
    return num;
}

__device__ void
over_select(Wv *wvar, int warpsize, int N,int overN, curandState local_state, gpu_graph *G, int *colcount, int source, int *Gmax, int bitflag)
{
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpID = tid/32;
    int warpTID= threadIdx.x%32;
    float *degree_l = wvar->degree;
    int *bitmap = wvar->bitmap;
    int *selected_list = wvar->selected;
    int neighbor_length =wvar->NL; 
    int neighbor_start= wvar->NS;
    int *total_counter = wvar->total_counter; 
    warpsize=32;
    int prefix=0, index=0;
    int new_neighbor;
    clock_t start_time,stop_time;
    float pref_time;
    int counter;
    wvar->sindex[0]=0;
    // decide if prefix sum is required
    if(neighbor_length>N) { prefix=1; }
    if(prefix==1)
    {
        start_time = clock();
        ITS(degree_l, 0, warpsize, neighbor_length);  
        stop_time =clock();
        pref_time= float(stop_time-start_time);
        index=warpTID;
        start_time= clock();  
        
        // reset bitmaps
        int start=warpTID;
        int end = neighbor_length/32 + 1;
        for(int i=start;i<end;i+=warpsize){bitmap[i]=0;}
        
        counter=0;
        int flag=1;
        while(flag){
            counter+=1;
            if(counter>4000){break;}
            // if(warpTID==0){printf("Iteration: %d\n",counter);}
            index=warpTID;
            while(index<overN)
            {
                int is_in=1, selected=0;
                colcount[index]=0;
                float r = curand_uniform(&local_state);
                selected= binary_search(0,neighbor_length,r,degree_l);
                // wvar->tempSelected[index]= selected;
                is_in= bitmap_search(bitmap,0,selected);
                if(is_in==0){
                    int pos= atomicAdd(&wvar->sindex[0],1);
                    if(pos<N){
                    selected_list[pos]= selected; 
                    // printf("index: %d,random number: %f, selected: %d, is_in: %d\n",index,r,selected,is_in);   
                    }
                }
                index+=warpsize;
            }
            __syncwarp();   
            if(wvar->sindex[0]<N){flag=1;}
            else{flag=0;break;}
            // printf("index: %d,random number: %f, selected: %d, is_in: %d\n",index,r,selected,is_in);        
            // new_neighbor= G->adj_list[selected+neighbor_start];
            // selected_list[index]= new_neighbor;                
            
        }    
        __syncwarp();
        stop_time= clock();
        float samp_time = float(stop_time-start_time);
        if(warpTID==0){
            total_counter[0]+=samp_time;
            // printf("%d, %d, %d, %d, %.0f, %.0f\n",counter,neighbor_length, N,source,pref_time,samp_time);
        }
    }
    else
    {
        index=warpTID;
        while(index<N)
        {
            new_neighbor = G->adj_list[index + neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
    }
}

__device__ void
select(Wv *wvar, Cp *cache, int N,int overN, curandState local_state, gpu_graph *G, int *colcount, int source, int *Gmax, int bitflag, int Fcache)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpID = tid/32;
    int warpTID= threadIdx.x%32;
    float *degree_l = wvar->degree;
    int *bitmap = wvar->bitmap;
    int *selected_list = wvar->selected;
    int neighbor_length =wvar->NL; 
    int neighbor_start= wvar->NS;
    int *total_counter = wvar->total_counter; 
    int warpsize=32;
    int prefix=0, index=0;
    int new_neighbor;
    clock_t start_time,stop_time;
    float pref_time;
    // if(source%2==0){N=N+1;}
    // decide if prefix sum is required
    if(neighbor_length>N) { prefix=1; }
    if(prefix==1)
    {
        start_time = clock();
        if(Fcache){
            int offset=G->beg_pos[source];
            if(cache->status[source]==1){
                // if(warpTID==0){printf("avoided.\n");}
                read_prob(degree_l,cache->probability,neighbor_length,offset);
            }
            else{
                ITS(degree_l, 0, warpsize, neighbor_length);
                write_prob(degree_l,cache->probability,neighbor_length,offset);
                if(warpTID==0){cache->status[source]=1;}
            }
        }
        else{ITS(degree_l, 0, warpsize, neighbor_length);}
        
        stop_time =clock();
        pref_time= float(stop_time-start_time);
        index=warpTID;
        while(index<neighbor_length){
            // if(warpID==2){printf("id:%d,value:%f\n",index,degree_l[index]);}
            degree_l[index];
            index+=warpsize;
        }    
        start_time= clock();  
        
        // reset bitmaps
        int start= warpTID;
        int end= neighbor_length/32 + 1;
        for(int i=start;i<end;i+=warpsize){bitmap[i]=0;}
        index=warpTID;
        while(index<N)
        {
            int is_in=1, selected=0;
            colcount[index]=0;
            while(is_in==1){
                colcount[index]+=1;
                float r = curand_uniform(&local_state);
                selected= binary_search(0,neighbor_length,r,degree_l);
                if(bitflag==1){is_in= bitmap_search(bitmap,0,selected);}
                else{is_in= linear_search(bitmap,0,selected);}
                // printf("index: %d,random number: %f, selected: %d, is_in: %d\n",index,r,selected,is_in);
                if(is_in==0){
                    new_neighbor= G->adj_list[selected+neighbor_start];
                    selected_list[index]= new_neighbor;
                    #ifdef profile
                    printf("Added %d to sampled.\n",selected);
                    #endif
                    
                    break;}
                if(colcount[index]>400){
                    selected_list[index]= 0;
                    break;
                }
            }
            // Add new neighbor to 
            // printf("Index: %d, count: %d\n",index,colcount[index]);
            index+=warpsize;
        }
        //
        //
        __syncwarp();
        stop_time= clock();
        float samp_time = float(stop_time-start_time);
        if(warpTID==0){
            int longer=max(colcount,N);
            atomicAdd(&Gmax[0], longer);
            total_counter[0]+=samp_time;
            // printf("%d, %d, %d, %d, %.0f, %.0f\n",longer,neighbor_length, N,source,pref_time,samp_time);
        }
    
    }
    else
    // pick up neighors
    {
        index=warpTID;
        while(index<N)
        {
            new_neighbor = G->adj_list[index + neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
    }
}

__device__ void
naive_ITS(Wv *wvar, Cp *cache, int N,int overN, curandState local_state, gpu_graph *G, int *colcount, int source, int *Gmax, int bitflag, int Fcache)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpID = tid/32;
    int warpTID= threadIdx.x%32;
    float *degree_l = wvar->degree;
    int *bitmap = wvar->bitmap;
    int *selected_list = wvar->selected;
    int neighbor_length =wvar->NL; 
    int neighbor_start= wvar->NS;
    int *total_counter = wvar->total_counter; 
    int warpsize=32;
    int prefix=0, index=0;
    int new_neighbor;
    clock_t start_time,stop_time;
    float pref_time;
    // if(source%2==0){N=N+1;}
    // decide if prefix sum is required
    if(neighbor_length>N) { prefix=1; }
    if(prefix==1)
    {
        start_time = clock();
        if(Fcache){
            int offset=G->beg_pos[source];
            if(cache->status[source]==1){
                // if(warpTID==0){printf("avoided.\n");}
                read_prob(degree_l,cache->probability,neighbor_length,offset);
            }
            else{
                ITS(degree_l, 0, warpsize, neighbor_length);
                write_prob(degree_l,cache->probability,neighbor_length,offset);
                if(warpTID==0){cache->status[source]=1;}
            }
        }
        else{ITS(degree_l, 0, warpsize, neighbor_length);}
        
        stop_time =clock();
        pref_time= float(stop_time-start_time);
        index=warpTID;
        while(index<neighbor_length){
            // if(warpID==2){printf("id:%d,value:%f\n",index,degree_l[index]);}
            degree_l[index];
            index+=warpsize;
        }    
        start_time= clock();  
        
        // reset bitmaps
        int start= warpTID;
        int end= neighbor_length/32 + 1;
        for(int i=start;i<end;i+=warpsize){bitmap[i]=0;}
        index=warpTID;
        while(index<N)
        {
            int is_in=1, selected=0;
            colcount[index]=0;
            while(is_in==1){
                colcount[index]+=1;
                float r = curand_uniform(&local_state);
                selected= binary_search(0,neighbor_length,r,degree_l);
                if(bitflag==1){is_in= bitmap_search(bitmap,0,selected);}
                else{is_in= linear_search(bitmap,0,selected);}
                // printf("index: %d,random number: %f, selected: %d, is_in: %d\n",index,r,selected,is_in);
                if(is_in==0){
                    new_neighbor= G->adj_list[selected+neighbor_start];
                    selected_list[index]= new_neighbor;
                    #ifdef profile
                    printf("Added %d to sampled.\n",selected);
                    #endif
                    
                    break;}
                if(colcount[index]>400){
                    selected_list[index]= 0;
                    break;
                }
            }
            // Add new neighbor to 
            // printf("Index: %d, count: %d\n",index,colcount[index]);
            index+=warpsize;
        }
        //
        if(N<4){
        ITS(degree_l, 0, warpsize, neighbor_length);
        float r= curand_uniform(&local_state);
        int selected= binary_search(0,neighbor_length,r,degree_l);
        }
        //
        __syncwarp();
        stop_time= clock();
        float samp_time = float(stop_time-start_time);
        if(warpTID==0){
            int longer=max(colcount,N);
            atomicAdd(&Gmax[0], longer);
            total_counter[0]+=samp_time;
            // printf("%d, %d, %d, %d, %.0f, %.0f\n",longer,neighbor_length, N,source,pref_time,samp_time);
        }
    
    }
    else
    // pick up neighors
    {
        index=warpTID;
        while(index<N)
        {
            new_neighbor = G->adj_list[index + neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
    }
}

__device__ void
normalize_over_select(Wv *wvar, int warpsize, int N, int overN, curandState local_state, gpu_graph *G, int *colcount, int source, int *Gmax)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpID = tid/32;
    int warpTID= threadIdx.x%32;
    float *degree_l = wvar->degree;
    int *bitmap = wvar->bitmap;
    int *selected_list = wvar->selected;
    int neighbor_length =wvar->NL; 
    int neighbor_start= wvar->NS;
    int *total_counter = wvar->total_counter; 
    warpsize=32;
    int prefix=0, index=0;
    int new_neighbor;
    clock_t start_time,stop_time;
    float pref_time;
    wvar->sindex[0]=0;
    if(neighbor_length>N) { prefix=1; }
    if(prefix==1)
    {
        start_time = clock();
        ITS(degree_l, 0, warpsize, neighbor_length);  
        stop_time =clock();
        pref_time= float(stop_time-start_time);
        index=warpTID;
        
               // reset bitmaps
               int start=warpTID;
               int end = neighbor_length/32 + 1;
               // if(warpTID==0)printf("Bitmap end:%d\n",end);
               for(int i=start;i<end;i+=warpsize)
               {
                   bitmap[i]=0;
                       //printf("Bitmap cleared at %d\n",i);
               }
               index=warpTID;
        while(index<overN)
        {
            int is_in=1, selected=0;
            colcount[index]=0;
            float lb=0,hb=1;
            float temp=0;
            float a,b;
            float r = curand_uniform(&local_state);
            while(is_in==1){
                colcount[index]+=1;
                selected= binary_search(0,neighbor_length,r,degree_l);
                is_in= bitmap_search(bitmap,0,selected);
                selected_list[index] = selected;
                // printf("index: %d,random number: %.2f, selected: %d, is_in: %d\n",index,r,selected,is_in);
                if(is_in==0){
                    int pos= atomicAdd(&wvar->sindex[0],1);
                    if(pos<N){
                    selected_list[pos]= selected; 
                    break;}}
                if(is_in==1){
                    float value = degree_l[selected];
                    if(r>value){
                        a= degree_l[selected];
                        b= degree_l[selected+1];}
                    else{
                        a= degree_l[selected-1];
                        b= degree_l[selected];}
                    // if(lb==a && hb==b){};
                    // float temp = 0.23;
                    temp= (float) (a-lb)/(a-lb+hb-b);
                    // printf("a: %.2f, b: %.2f,lb: %.2f, hb: %.2f, temp: %.2f\n",a,b,lb,hb,temp);
                    if(r< temp)
                    {   
                        // printf("Update hb.\n");
                        r= (lb+r*(a-lb)); 
                        hb=a;}
                    else
                    {
                        // printf("Update lb.\n");
                        r= (b+(hb-b)*r);
                        lb=b;}
                    // printf("\nNew r: %.2f, lb: %.2f, hb: %.2f\n",r,lb,hb);
                }
                
                if(colcount[index]>80){break;}
                // else{
                //     // atomicAdd(&colcount[0],1);
                //     printf("Repeated. Index: %d, selected: %d\n",index,selected);
                //     }
            }
            new_neighbor= G->adj_list[selected+neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
        __syncwarp();
        stop_time= clock();
        float samp_time = float(stop_time-start_time);
        if(warpTID==0){
            int longer=max(colcount,N);
            total_counter[0]+=samp_time;
       }
        
    }
    else{
        index=warpTID;
        while(index<N)
        {
            new_neighbor = G->adj_list[index + neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
    }
}



__device__ void
normalize(Wv *wvar, int warpsize, int N, int overN, curandState local_state, gpu_graph *G, int *colcount, int source, int *Gmax)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpID = tid/32;
    int warpTID= threadIdx.x%32;
    float *degree_l = wvar->degree;
    int *bitmap = wvar->bitmap;
    int *selected_list = wvar->selected;
    int neighbor_length =wvar->NL; 
    int neighbor_start= wvar->NS;
    int *total_counter = wvar->total_counter; 
    warpsize=32;
    int prefix=0, index=0;
    int new_neighbor;
    clock_t start_time,stop_time;
    float pref_time;
    if(neighbor_length>N) { prefix=1; }
    if(prefix==1)
    {
        start_time = clock();
        ITS(degree_l, 0, warpsize, neighbor_length);  
        stop_time =clock();
        pref_time= float(stop_time-start_time);
        index=warpTID;
        
               // reset bitmaps
               int start=warpTID;
               int end = neighbor_length/32 + 1;
               // if(warpTID==0)printf("Bitmap end:%d\n",end);
               for(int i=start;i<end;i+=warpsize)
               {
                   bitmap[i]=0;
                       //printf("Bitmap cleared at %d\n",i);
               }
               index=warpTID;
        while(index<N)
        {
            int is_in=1, selected=0;
            colcount[index]=0;
            float lb=0,hb=1;
            float temp=0;
            float a,b;
            float r = curand_uniform(&local_state);
            while(is_in==1){
                colcount[index]+=1;
                selected= binary_search(0,neighbor_length,r,degree_l);
                is_in= bitmap_search(bitmap,0,selected);
                // selected_list[index] = selected;
                // printf("index: %d,random number: %.2f, selected: %d, is_in: %d\n",index,r,selected,is_in);
                if(is_in==0){break;}
                if(is_in==1){
                    float value = degree_l[selected];
                    if(r>value){
                        a= degree_l[selected];
                        b= degree_l[selected+1];}
                    else{
                        a= degree_l[selected-1];
                        b= degree_l[selected];}
                    // if(lb==a && hb==b){};
                    // float temp = 0.23;
                    temp= (float) (a-lb)/(a-lb+hb-b);
                    // printf("a: %.2f, b: %.2f,lb: %.2f, hb: %.2f, temp: %.2f\n",a,b,lb,hb,temp);
                    if(r< temp)
                    {   
                        // printf("Update hb.\n");
                        r= (lb+r*(a-lb)); 
                        hb=a;}
                    else
                    {
                        // printf("Update lb.\n");
                        r= (b+(hb-b)*r);
                        lb=b;}
                    // printf("\nNew r: %.2f, lb: %.2f, hb: %.2f\n",r,lb,hb);
                }
                
                if(colcount[index]>1000){break;}
                // else{
                //     // atomicAdd(&colcount[0],1);
                //     printf("Repeated. Index: %d, selected: %d\n",index,selected);
                //     }
            }
            new_neighbor= G->adj_list[selected+neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
        __syncwarp();
        stop_time= clock();
        float samp_time = float(stop_time-start_time);
        if(warpTID==0){
            int longer=max(colcount,N);
            // printf("%d, %d, %d, %d, %.0f, %.0f\n",longer,neighbor_length, N,source,pref_time,samp_time);
            total_counter[0]+=samp_time;
       }
        
    }
    else{
        index=warpTID;
        while(index<N)
        {
            new_neighbor = G->adj_list[index + neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
    }
}

__device__ void
heur_normalize(Wv *wvar, Cp *cache, int N, int overN, curandState local_state, gpu_graph *G, int *colcount, int source, int *Gmax,int bitflag,int Fcache)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpID = tid/32;
    int warpTID= threadIdx.x%32;
    float *degree_l = wvar->degree;
    int *bitmap = wvar->bitmap;
    int *selected_list = wvar->selected;
    int neighbor_length =wvar->NL; 
    int neighbor_start= wvar->NS;
    int *total_counter = wvar->total_counter; 
    int warpsize=32;
    int prefix=0, index=0;
    int new_neighbor;
    clock_t start_time,stop_time;
    float pref_time;
    // For forest fire
    // if(source%2==0){N=N+1;}
    if(neighbor_length>N) { prefix=1; }
    if(prefix==1)
    {
        start_time = clock();
        if(Fcache){
            int offset=G->beg_pos[source];
            if(cache->status[source]==1){
                // if(warpTID==0){printf("avoided.\n");}
                read_prob(degree_l,cache->probability,neighbor_length,offset);
            }
            else{
                ITS(degree_l, 0, warpsize, neighbor_length);
                write_prob(degree_l,cache->probability,neighbor_length,offset);
                if(warpTID==0){cache->status[source]=1;}
            }
        }
        else{ITS(degree_l, 0, warpsize, neighbor_length);}
        stop_time =clock();
        pref_time= float(stop_time-start_time);
        index=warpTID;
        
               // reset bitmaps
               int start=warpTID;
               int end = neighbor_length/32 + 1;
               // if(warpTID==0)printf("Bitmap end:%d\n",end);
               for(int i=start;i<end;i+=warpsize)
               {
                   bitmap[i]=0;
                       //printf("Bitmap cleared at %d\n",i);
               }
               index=warpTID;
        while(index<N)
        {
            int is_in=1, selected=0;
            colcount[index]=0;
            float lb=0,hb=1;
            float temp=0;
            float a,b;
            
            while(is_in==1){
                float r = curand_uniform(&local_state);  
                int localCount=0;
                colcount[index]+=1;
                selected= binary_search(0,neighbor_length,r,degree_l);
                is_in= bitmap_search(bitmap,0,selected);
                    // printf("index: %d,random number: %.2f, selected: %d, is_in: %d\n",index,r,selected,is_in);
                if(is_in==0){
                        new_neighbor= G->adj_list[selected+neighbor_start];
                        selected_list[index]= new_neighbor;
                        break;
                }
                if(is_in==1){
                        float value = degree_l[selected];
                        if(r>value){
                            a= degree_l[selected];
                            b= degree_l[selected+1];}
                        else{
                            a= degree_l[selected-1];
                            b= degree_l[selected];}
                        // if(lb==a && hb==b){};
                        // float temp = 0.23;
                        float lambda= (float) (a-lb)/(a-lb+hb-b);
                        float delta= (float) (b-a)/(hb-lb);
                        r= (float) r/lambda;
                        // printf("a: %.2f, b: %.2f,lb: %.2f, hb: %.2f, temp: %.2f\n",a,b,lb,hb,temp);
                        if(r< a)
                        {   
                            hb=a;}
                        else
                        {r= r + delta;
                            lb=b;}
                        // printf("index: %d,random number: %.2f, selected: %d, is_in: %d\n",index,r,selected,is_in);
                        localCount+=1;
                    
                    selected= binary_search(0,neighbor_length,r,degree_l);
                    is_in= bitmap_search(bitmap,0,selected);
                    if(is_in==0){
                        new_neighbor= G->adj_list[selected+neighbor_start];
                        selected_list[index]= new_neighbor;
                        break;
                    }
                }
                if(colcount[index]>400){break;}
            }
            index+=warpsize;
        }
        __syncwarp();
        stop_time= clock();
        float samp_time = float(stop_time-start_time);
        if(warpTID==0){
            int longer=max(colcount,N);
            atomicAdd(&Gmax[0], longer);
            // printf("%d, %d, %d, %d, %.0f, %.0f\n",longer,neighbor_length, N,source,pref_time,samp_time);
            // total_counter[0]+=samp_time;
       }
        
    }
    else{
        index=warpTID;
        while(index<N)
        {
            new_neighbor = G->adj_list[index + neighbor_start];
            selected_list[index]= new_neighbor;
            index+=warpsize;
        }
    }
}



__device__ int
get_neighbors(gpu_graph *graph, int vertex, Wv *wvar, int VertCount){
    int warpTID= threadIdx.x%32;
    int index= warpTID;
    int len= graph->degree_list[vertex];
    wvar->NL=len;
    int neighbor_start=graph->beg_pos[vertex];
    wvar->NS= neighbor_start;
    #ifdef profile
    // if(warpTID==0){printf("Source: %d, NLen: %d, Nstart: %d\n",vertex, len,neighbor_start);}
    #endif
    while(index<len)
    {
        int neighbor= graph->adj_list[neighbor_start + index];
        wvar->neighbors[index]= neighbor;
        wvar->degree[index]= EdgeBias(neighbor,graph);  
        // {printf("Neighbor:%d, tid:%d\n",wvar->neighbors[index],index);}
        index+=warpSize;
    }
    return len; 
}

// __device__ void
// next(S){

// }

__device__ int 
linear_search(int neighbor,int *partition1, int *bin_count, int bin, int BIN_SIZE, int BUCKETS)
{
    if(bin>=32){printf("Bin error.\n");}
	int len = bin_count[bin];	
	int i = bin;
	// printf("\nL: %d, I:%d\n",len,i);
	int step=0;
	while(step<len)
	{
		// #ifdef profile
		// atomicAdd(&counter[0],1);
		// #endif
		int test=partition1[i];
		// printf("Neighbor: %d, Test: %d, address: %d\n",neighbor,test,i);
		if(test==neighbor)
		{
			//printf("Duplicate detected -------------------------------------------------------\n");
			return 1;
		}
		else
		{
			i+=BUCKETS;
		}
		step+=1;
	}
	return 0;
}


__device__ int
duplicate(Ht *hashtable, int vertex){
    int BUCKETS = hashtable->BUCKETS;
    int bin= vertex % BUCKETS;		
    int BIN_SIZE = hashtable->bin_size;
    // #ifdef profile
    // printf("Bucket %d, bin: %d\n",BUCKETS,bin);
    // #endif
    int is_in=linear_search(vertex,hashtable->hash,hashtable->bin_counter,bin,BIN_SIZE,BUCKETS);
		// 	// if(is_in==1){printf("Duplicated Found: %d\n",new_neighbor);}
	return is_in;
    }

__device__ void
add_hash(Ht *hashtable, int vertex)
{
    int BUCKETS = hashtable->BUCKETS;
    int bin= vertex % BUCKETS;		
    int index=atomicAdd(&hashtable->bin_counter[bin],1);
    if(index>100){printf("error. %d\n",index);}
    #ifdef profile
    printf("Add: %d, bin: %d, INdex: %d\n",vertex,bin,index);
    #endif
    hashtable->hash[index*BUCKETS+ bin]=vertex;
}



__device__ int
linear_duplicate(Si *samples, int vertex){
    int warpTID=threadIdx.x%32;
    int index= warpTID;
    while(index<samples->start[0])
    {
        if(vertex==samples->edge[index]){
            return 1;
            break;
        }
    }
    return 0;
}

__device__ void
frontier(gpu_graph *G,Sampling *S, int warpId,int SampleID, int N, int source, int sourceIndex, int hash, int Depth)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpTID= threadIdx.x%32;
    int *selected=S->wvar[warpId].selected;
    int index=warpTID;
    int is_in=0;
    while(index<N)
    {   
        // if(threadIdx.x==0){printf("Depth: %d\n",Depth);}
        int vertex= Update(G,selected[index], source);
        // if(hash){is_in= duplicate(&S->hashtable[SampleID], vertex);}
        // else{is_in= linear_duplicate(&S->samples[SampleID], vertex);}
        int pos=atomicAdd(&S->samples[SampleID].start[0],1);
        // total count
        atomicAdd(&S->sampled_count[0],1); 
        #ifdef profile
        
        // printf("Added to sampled.\n SID: %d, Updated: %d, pos: %d, is_in: %d\n",SampleID,vertex,pos,is_in);
        #endif
		S->samples[SampleID].vertex[pos]=source;
        S->samples[SampleID].edge[pos]=vertex;
		if(is_in==0)
		{
            // add_hash(&S->hashtable[SampleID], vertex);
            int currDepth= S->candidate.depth[sourceIndex];
            if(currDepth < (Depth-1)){
        // #ifdef profile
        // printf("warpID: %d, Curr:%d, Added %d to queue.\n",tid/32,currDepth,vertex);
        // #endif
                int Qid= atomicAdd(&S->candidate.end[0],1);
                S->candidate.vertices[Qid]= vertex;
                S->candidate.instance_ID[Qid]= S->candidate.instance_ID[sourceIndex];
                S->candidate.depth[Qid]= currDepth+1;
            }
        }
        index+=warpSize;
    }    
    // __syncwarp();
}


__device__ int
ITS_MDRW(Wv *wvar,curandState local_state, gpu_graph *G, int neighbor_length, float r)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int warpID = tid/32;
    int warpTID= threadIdx.x%32;
    float *degree_l = wvar->degree;
    int neighbor_start= wvar->NS;
    int *total_counter = wvar->total_counter; 
    int warpsize=32;
    int prefix=0, index=0;
    int new_neighbor;
    clock_t start_time,stop_time;
    float pref_time;
    // if(source%2==0){N=N+1;}
    // decide if prefix sum is required
    if(neighbor_length>1) { prefix=1; }
    if(prefix==1)
    {
        start_time = clock();
        ITS(degree_l, 0, warpsize, neighbor_length);
        __syncwarp();
        #ifdef profile
        if(threadIdx.x==0){
            for(int i=0;i<neighbor_length;i+=1)
            {
                printf("%.2f,\t",degree_l[i]);
            }
            printf("\n");}
        #endif
        stop_time =clock();
        pref_time= float(stop_time-start_time);
        index=warpTID;
        int selected=0;
        selected= binary_search(0,neighbor_length,r,degree_l);
        #ifdef profile
        if(warpTID==0){printf("Random: %.2f, selected: %d\n",r,selected);}
        #endif
        new_neighbor= G->adj_list[selected+neighbor_start];
        return selected;
    }
    else
    {
        return 0;
        // new_neighbor = G->adj_list[neighbor_start];
    }
}


#endif


