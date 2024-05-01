#pragma once
#include <ATen/cuda/MyTensorSync.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Logging.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cassert>
namespace at {
using TensorId = void*;
class Tensor;

void TORCH_CUDA_CPP_API recordAndReplaceEvent(
    at::Tensor& src,
    const at::Tensor& new_tensor,
    c10::DeviceIndex recordAtDeviceIdx,
    at::cuda::CUDAStream stream);

void TORCH_CUDA_CPP_API recordEvent(
    TensorId srcId,
    TensorId dstId,
    c10::DeviceIndex recordAtDeviceIdx,
    at::cuda::CUDAStream stream);

void TORCH_CUDA_CPP_API syncEvent(
    TensorId ptr,
    c10::DeviceIndex syncAtDeviceIdx,
    c10::DeviceIndex currentTensorDeviceIdx,
    at::cuda::CUDAStream stream,
    bool enableLog = true);

std::string TORCH_CUDA_CPP_API pointerInfo(TensorId ptr);

void TORCH_CUDA_CPP_API initDeviceEventManager(int deviceCount);

} // namespace at