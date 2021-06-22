#include "fake_quantize.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<Tensor> fake_quantize(Tensor a, int bit_width,int type,int c,int aciq){
  CHECK_INPUT(a);
  return fake_quantize_cuda(a, bit_width,type,c,aciq);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("fake_quantize", &fake_quantize, "NCNN Fake Quantization (CUDA)");
}






