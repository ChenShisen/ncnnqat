#include "fake_quantize.h"


__global__ void max_reduce(float* __restrict__ data,float* out_ptr,int width,int lg_n) //preset data[i] >=0 
{
    __shared__ float* middleware[blockSize];                                
    const float min_positive_float = 1e-6;                                       
    int row = blockIdx.x * width + threadIdx.x;                             
    int bid = blockIdx.x;                                                   
    int tid = threadIdx.x;                                                  
    int tid_tmp = threadIdx.x;                                              

    //if(tid<width) {middleware[tid] = &(data[row]);}                       
    if(tid<width) middleware[tid] = data+row;                               
    else middleware[tid] = &min_positive_float;                             
    row+=blockSize;                                                         
    tid_tmp +=blockSize;                                                    
    while(tid_tmp<width)                                                    
    {
        //if(data[row]>*(middleware[tid])) middleware[tid] = &(data[Row]);
	if(fabs(data[Row])>fabs(*(middleware[tid]))) middleware[tid] = data+row;
	row+=blockSize;                                                     
	tid_tmp +=blockSize;
    }
    __syncthreads();                                                        
	
    //for(int i=blockSize/2; i>0; i/=2)
    for(int i=lg_n/2; i>0; i/=2)                                           
    {
        if(tid<i)
	{
	    if(fabs(*(middleware[tid+i]))>fabs(*(middleware[tid]))) middleware[tid]=middleware[tid+i];
	}
        __syncthreads();
    }
	
    if(tid==0) out_ptr[bid] = fabs(*(middleware[0]));                             	
}
__global__ void fake_quantize_layer_google(float* __restrict__ a,
                                           float* o,  
                                           float* o1,  
                                           float* movMax,  
					   int size,
					   int bit_width,
					   float* max_entry) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
    {
	const float momenta = 0.95; 
	float mov_max_tmp = movMax[0];
	if(mov_max_tmp<1e-6) mov_max_tmp=fabs(*max_entry);  //movMax dafault 0 ,now first step set it a non zero data
	else  mov_max_tmp= mov_max_tmp * momenta + fabs(*max_entry) * (1.-momenta);  // #GOOGLE QAT : movMax = movMax*momenta + max(abs(tensor))*(1-momenta)    momenta = 0.95
        float data_scale = __powf(2.,bit_width-1.)-1;
		
	float scale;
        if(mov_max_tmp < 1e-6) scale =  __fdividef(data_scale,1e-6);
	else scale =  __fdividef(data_scale,mov_max_tmp);
			
	int o_int = round(a[index]*scale);
	//o[index] = __fdividef(round(a[index]*scale),scale);
	if(o_int>data_scale) o_int=(int)data_scale;
	else if(o_int<-data_scale) o_int=(int)(-data_scale);
	else {};
	o[index] =  __fdividef(o_int*1.,scale);
		
	if(index==0) 
	{
	    o1[0] = scale;
	    movMax[0] = mov_max_tmp;
	}
    }
}


__global__ void fake_quantize_layer_aciq(float* __restrict__ a,
                                         float* o,  
                                         float* o1,  
                                         float* movMax,
                                         int feature_pixl_num,											
					 int size,
					 int bit_width,
					 float* max_entry) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
    {
        const float momenta = 0.95; 
        float mov_max_tmp = movMax[0];
        if(mov_max_tmp<1e-6) mov_max_tmp=fabs(*max_entry);  //movMax dafault 0 ,now first step set it a non zero data
        else  mov_max_tmp= fabs(*max_entry);//mov_max_tmp * momenta + fabs(*max_entry) * (1.-momenta);  // #GOOGLE QAT : movMax = movMax*momenta + max(abs(tensor))*(1-momenta)    momenta = 0.95
        float data_scale = __powf(2.,bit_width-1.)-1;
		
        const float alpha_gaussian[8] = {0, 1.71063519, 2.15159277, 2.55913646, 2.93620062, 3.28691474, 3.6151146, 3.92403714};
        const double gaussian_const = (0.5 * 0.35) * (1 + sqrt(3.14159265358979323846 * __logf(4.)));
        double std = (mov_max_tmp * 2 * gaussian_const) / sqrt(2 * __logf(feature_pixl_num));
        float threshold = (float)(alpha_gaussian[bit_width - 1] * std);
		
        float scale;
        if(threshold < 1e-6) scale =  __fdividef(data_scale,1e-6);
        else scale =  __fdividef(data_scale,threshold);
	//float o_index = __fdividef(round(a[index]*scale),scale);
	int o_int = round(a[index]*scale);
	//o[index] = __fdividef(round(a[index]*scale),scale);
	if(o_int>data_scale) o_int=(int)data_scale;
	else if(o_int<-data_scale) o_int=(int)(-data_scale);
	else {};
	o[index] =  __fdividef(o_int*1.,scale);
		
	if(index==0) 
	{
	    o1[0] = scale;
	    movMax[0] = mov_max_tmp;
	}
    }
}

__global__ void fake_quantize_channel_aciq(float* __restrict__ a,
                                           float* o,  
                                           float* o1,  
					   int size,
					   int bit_width,
					   float* max_entry_arr, //max_entry_arr already>0
					   int channel_num) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
	int channel = index/channel_num;
	float* max_entry = max_entry_arr+channel;
	float data_scale = __powf(2.,bit_width-1.)-1;
        if((*max_entry) < 1e-6)
	{
	    //if(index%channel_num==0) o1[channel] = scale;
	    *max_entry = 1e-6;
            //return;
        }
	const float alpha_gaussian[8] = {0, 1.71063519, 2.15159277, 2.55913646, 2.93620062, 3.28691474, 3.6151146, 3.92403714};
	const double gaussian_const = (0.5 * 0.35) * (1 + sqrt(3.14159265358979323846 * __logf(4.)));
	double std = ((*max_entry) * 2 * gaussian_const) / sqrt(2 * __logf(channel_num));
	float threshold = (float)(alpha_gaussian[bit_width - 1] * std);
		
	float scale =  __fdividef(data_scale,threshold);
	int o_int = round(a[index]*scale);
	if(o_int>data_scale) o_int=(int)data_scale;
	else if(o_int<-data_scale) o_int=(int)(-data_scale);
	else {};
	o[index] = __fdividef(o_int*1.,scale);
	if(index%channel_num==0) o1[channel] = scale;
    }
}
__global__ void fake_quantize_channel_cuda(float* __restrict__ a,
                                           float* o,  
                                           float* o1,  
					   int size,
					   int bit_width,
					   float* max_entry_arr, //max_entry_arr already>0
					   int channel_num) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
    {
	int channel = index/channel_num;
	float* max_entry = max_entry_arr+channel;
	float data_scale = __powf(2.,bit_width-1.)-1;
        if((*max_entry) < 1e-6)
	{
	    //if(index%channel_num==0) o1[channel] = scale;
	    *max_entry = 1e-6;
            //return;
        }
	float scale =  __fdividef(data_scale,*max_entry);
	o[index] = __fdividef(round(a[index]*scale),scale);
	if(index%channel_num==0) o1[channel] = scale;
    }
}
std::vector<Tensor> fake_quantize_activate_cuda(Tensor a, int bit_width ,int aciq)
{
    auto o = at::zeros_like(a); //q out
    auto o1 = at::zeros({1}, a.options());  //scale
    auto movMax = at::zeros({1}, a.options());  //max of tensor  #GOOGLE QAT  movMax = movMax*momenta + max(abs(tensor))*(1-momenta)    momenta = 0.95
    int64_t size = a.numel();
	
    int batch_size = a.size(0);//batchsize
    int feature_pixl_num = size/batch_size;
    
    Tensor max_entry = at::max(at::abs(a));
    int blockNums = (size + blockSize - 1) / blockSize;
	
    if(aciq==0) //movmax
    {
	//printf("layer_max....");
	fake_quantize_layer_google<<<blockNums, blockSize>>>(a.data_ptr<float>(),
							     o.data_ptr<float>(),
							     o1.data_ptr<float>(),
							     movMax.data_ptr<float>(),
							     size,
							     bit_width,
							     max_entry.data_ptr<float>());
    }
    else // aciq
    {
	//printf("layer_aciq....");
	fake_quantize_layer_aciq<<<blockNums, blockSize>>>(a.data_ptr<float>(),
							   o.data_ptr<float>(),
							   o1.data_ptr<float>(),
							   movMax.data_ptr<float>(),
							   feature_pixl_num,
							   size,
							   bit_width,
							   max_entry.data_ptr<float>());
    }
    return {o,o1,movMax};
}


std::vector<Tensor> fake_quantize_weight_cuda(Tensor a, int bit_width,int c ,int aciq) 
{
    auto o = at::zeros_like(a); //q out
    auto o1 = at::zeros({c}, a.options());  //scale
    int64_t size = a.numel();
    
    int blockNums = (size + blockSize - 1) / blockSize;
    int channel_num = size/c;
    auto max_entry_arr = at::zeros({c}, a.options());
	
    int lg_n = ceil(log2(channel_num*1.)); //2^x - channel_num >0 
    lg_n = pow(2,lg_n); //2^x
    if(lg_n>blockSize) lg_n=blockSize; //
	
    max_reduce <<<c, blockSize >>> (a.data_ptr<float>(),
				    max_entry_arr.data_ptr<float>(),
				    channel_num,
				    lg_n); //c block , each block get a max value
	
    if(aciq==0)
    {
	//printf("weight_max....");
	fake_quantize_channel_cuda<<<blockNums, blockSize>>>(a.data_ptr<float>(),
							     o.data_ptr<float>(),
							     o1.data_ptr<float>(),
							     size,
							     bit_width,
							     max_entry_arr.data_ptr<float>(),  //max_entry_arr already>0
							     channel_num);
    }
    else
    {
	//printf("weight_aciq....");
	fake_quantize_channel_aciq<<<blockNums, blockSize>>>(a.data_ptr<float>(),
							     o.data_ptr<float>(),
							     o1.data_ptr<float>(),
							     size,
							     bit_width,
							     max_entry_arr.data_ptr<float>(),  //max_entry_arr already>0
							     channel_num);	

    }		
    return {o,o1};
  }


std::vector<Tensor> fake_quantize_cuda(Tensor a, int bit_width,int type,int c,int aciq) 
{
    /*
    https://arxiv.org/pdf/1806.08342.pdf  2.5
    For weights,we use the actual minimum and maximum values to determine the quantizer parameters. 
    For activations, we use the moving average of the minimum and maximum values across batches to determine the quantizer parameters.
    float 6 7 ,double 15 16
    */
    if(type==0) return fake_quantize_activate_cuda(a,bit_width,aciq); //type==0 per layer	
    else return fake_quantize_weight_cuda(a,bit_width,c,aciq); //type==1 perchannel
}



