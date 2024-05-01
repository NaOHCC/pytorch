#pragma once

#include <c10/core/Allocator.h>
#include <ATen/cuda/CachingHostAllocator.h>

namespace at::cuda {

inline TORCH_CUDA_CPP_API at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}

inline TORCH_CUDA_CPP_API at::Allocator* getMyCUDAHostAllocator() {
  return getMyCachingCUDAHostAllocator();
}

} // namespace at::cuda
