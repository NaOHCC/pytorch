#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_debug_has_internal_overlap_native.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/pin_memory_native.h>
#endif

namespace at::native {

// Exposes at::has_internal_overlap as an operator for testing purposes
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

// Technically, we could force backends to explicitly say "no, we don't support
// pinned memory, always return false", but this makes life a little easier when
// you haven't loaded the backend extension at all (which can happen, e.g., on a
// CPU build of PyTorch and you try to check if something is CUDA pinned)
bool is_pinned_default(const Tensor& self, c10::optional<Device> device) {
  return false;
}

Tensor pin_memory(const Tensor& self, c10::optional<Device> device) {
  // Kind of mad that I have to do two dynamic dispatches here, pretty
  // annoying
  if (self.is_pinned(device)) {
    return self;
  }
  return at::_pin_memory(self, device);
}

// TODO: need modify
// 将tensor转移到pin memory, 目标设备改为GPU
Tensor my_cuda_host(const Tensor& self) {
  // model 目标设备是GPU就分配
  if (self.is_pinned() && self.device().is_cuda()) {
    TORCH_WARN("Already '", self.toString(), "' is pinned");
    return self;
  }
  auto* allocator = detail::getCUDAHooks().getMyCUDAHostAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);

  auto specified_options = self.options();
  specified_options = specified_options.device(at::kCUDA);

  auto tensor = at::empty({0}, specified_options)
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace at::native
