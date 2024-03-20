#ifndef THC_TENSOR_SYNC_INC
#define THC_TENSOR_SYNC_INC

#include <c10/cuda/CUDAStream.h>
#include <ATen/core/Tensor.h>

TORCH_CUDA_CPP_API void
recordEvent(at::Tensor& src, const at::Tensor& dst, int device, at::cuda::CUDAStream stream);

TORCH_CUDA_CPP_API void
syncEvent(void* ptr, int device, int cur_device, at::cuda::CUDAStream stream);

TORCH_CUDA_CPP_API void
initEventManager(int device_count);

#endif
