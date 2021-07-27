#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>
#include <tuple>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <vector>

const int blockSize = 1024;
//#define blockSize 1024

using namespace at;



std::vector<Tensor> fake_quantize_cuda(Tensor a, int bit_width=8,int type=1,int c=1,int aciq=0);

std::vector<Tensor> fake_quantize_activate_cuda(Tensor a, int bit_width ,int aciq);
std::vector<Tensor> fake_quantize_weight_cuda(Tensor a, int bit_width,int c,int aciq);


__global__ void max_reduce(float* __restrict__ data,float* out_ptr,int width,int lg_n);




__global__ void fake_quantize_layer_google(float* __restrict__ a,
					   float* o,  
                                           float* o1,
					   float* movMax,
				           int size,
					   int bit_width,
					   float* max_entry);
__global__ void fake_quantize_layer_aciq(float* __restrict__ a,
					 float* o,  
                                         float* o1,
					 float* movMax,
				         int feature_pixl_num,
					 int size,
					 int bit_width,
					 float* max_entry);

__global__ void fake_quantize_channel_cuda(float* __restrict__ a,
                                           float* o,  
                                           float* o1,  
					   int size,
				           int bit_width,
					   float* max_entry_arr,
					   int channel_num);
__global__ void fake_quantize_channel_aciq(float* __restrict__ a,
                                           float* o,  
                                           float* o1,  
					   int size,
					   int bit_width,
					   float* max_entry_arr,
					   int channel_num);

