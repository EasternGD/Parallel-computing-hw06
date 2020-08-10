

double euclidean(__global const  double *v1,const uint index_1, __global const double *v2,const uint index_2, const uint len) {
    // std::cout << "\nREMEBER TO IMPLEMENT EUCLIDEAN DISTANCE MEASURE " << __LINE__ << "@" << __FILE__;
    double ans = 0;
    for (uint i = 0; i < len; i++)
    {
        ans += (v1[i + index_1] - v2[index_2 + i]) * (v1[i + index_1] - v2[index_2 + i]);
    }
    ans = sqrt(ans);
    
    return ans;
   
}

double manhattan(__global const  double *v1,const uint index_1, __global const double *v2,const uint index_2, const uint len) {
	// you know what to do ...
    double ans = 0;
    for (size_t i = 0; i < len; i++)
    {
        ans += fabs(v1[i + index_1] - v2[index_2 + i]);
    }
    return ans;

}

double chebyshev(__global const  double *v1,const uint index_1, __global const double *v2,const uint index_2, const uint len) {
	// you know what to do ...
    double ans = 0;
    for (size_t i = 0; i < len; i++)
    {
        if (fabs(v1[i + index_1] - v2[index_2 + i]) > ans)
        {
            ans = fabs(v1[i + index_1] - v2[index_2 + i]);
        }
    }
    return ans;
}
/* 
// 
double minkowski(const double *v1, const double *v2, const size_t len) {
	return 0.0;
}
*/ 
double cosineSimilarity(__global const  double *v1,const uint index_1, __global const double *v2,const uint index_2, const uint len) {
	// you know what to do ...
	double dot = 0;
    double length_v1 = 0;
    double length_v2 = 0;
    for (size_t i = 0; i < len; i++)
    {
        dot += v1[i + index_1] * v2[index_2 + i];
        length_v1 += v1[i + index_1] * v1[i + index_1];
        length_v2 += v2[index_2 + i] * v2[index_2 + i];
    }
    double ans = acos(dot / (sqrt(length_v1 * length_v2))) /M_PI ;
    return ans;
}

// Write required kernel functions here (you need to write quite a few)...

__kernel
void getTrainningDataVector( __global double *in, uint in_len, __global double *out, uint out_len, uint iter, uint n)
{
    uint i = get_global_id(0);
    uint j = get_local_id(0);
    uint first_stage = iter * (in_len - out_len); 
    uint final_stage = iter * (in_len - out_len) + (in_len - out_len); 
        // printf("%d\t~\t%d\n",first_stage,final_stage);
        // printf("%d,%d\n",i,j);
    if(i < first_stage) 
    {
        out[i] = in[i];
        // printf("%d\n",i);
    }
    else if(i >= final_stage)
    {
        out[i - (in_len - out_len)] = in[i];
        // printf("%d\n",i - (in_len - out_len));
    }
    else
    {
        return;
    }
}

__kernel
void getValidationDataVector( __global double *in, uint in_len, __global double *out, uint out_len, uint iter, uint n)
{
    uint i = get_global_id(0);
    uint first_stage = iter * out_len; 
    uint final_stage = iter * out_len + out_len; 
        // printf("%d\t~\t%d\n",i,final_stage);
        // printf("%d\n",i);
    if(i >= first_stage && i < final_stage) 
    {
        out[i - iter * out_len] = in[i];
        // printf("%d\n",i - iter * out_len);
    }
    else
    {
        return;
    }
}


//v1 vali 5 v2 train 25
__kernel
void getDistanceMatrix(uint dev_len, __global double *v1, uint v1_len, __global double *v2, uint v2_len, __global double *distance, uint method)
{
    if(get_local_id(0) >= (v2_len/dev_len) * (v1_len/dev_len))
    {
        return;
    }

    // printf("%d",get_global_id(0));
    // __local double shmem[256];
    // uint localID = get_local_id(0);
    // uint i = localID + (v2_len/dev_len) * get_group_id(0);
    //load data from global memory into local shared memory
    // printf("%d",(v2_len/dev_len * get_group_id(0));
    // if(localID < dev_len )
    // {
        // shmem[localID] = v1[(i/(v2_len/dev_len)*dev_len) + localID];
        // printf("[%d]",localID);
        // printf("[%f]\n",v1[(i/(v2_len/dev_len)*dev_len) + localID]);
    // }
    // printf("[%d]",localID);
    // printf("[%f, %f, %f]\n",shmem[0],shmem[1],shmem[2]);
    // barrier(CLK_LOCAL_MEM_FENCE);
    uint i = get_global_id(0);
    uint index_1 = i/(v2_len/dev_len)*dev_len;
    uint index_2 = (i * dev_len) % v2_len;
    // printf("[%d]",index_2);
    switch (method)
    {
    case 0:
        distance[i] = euclidean(v1, index_1, v2, index_2, dev_len);
        break;
    case 1:
        distance[i] = manhattan(v1,index_1, v2, index_2, dev_len);
            break;
    case 2:
        distance[i] = chebyshev(v1,index_1, v2, index_2, dev_len);
        break;
    case 4:
        distance[i] = cosineSimilarity(v1,index_1, v2, index_2, dev_len);
        break;
    }

    // printf("[%f]\n",shmem[localID%3]);
    // printf("[%d][%d][%d]\n",index_1,index_2,dev_len);
        
    
}