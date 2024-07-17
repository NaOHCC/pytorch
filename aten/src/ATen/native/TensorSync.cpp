#include <ATen/ATen.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/cuda/CUDAStream.h>

namespace at {
namespace native {

// tensor 是device上的tensor
void my_record_and_replace_tensor(
    Tensor& self,
    const Tensor& new_tensor,
    Device recordAtDeviceIdx) {
  void* old_ptr = self.data_ptr();
  void* new_ptr = new_tensor.data_ptr();

  if (self.is_cuda() && self.is_pinned())
    return;
  if (new_ptr == old_ptr) {
    return;
  }

  c10::cuda::CUDAStream stream = detail::getCUDAHooks().current_stream();
  detail::getCUDAHooks().my_recordAndReplaceEvent(
      self, new_tensor, recordAtDeviceIdx.index(), stream);
}

void my_force_record_and_replace_tensor(
    Tensor& self,
    const Tensor& new_tensor,
    Device recordAtDeviceIdx,
    bool forceSync) {
  void* src_ptr = self.data_ptr();
  void* dst_ptr = new_tensor.data_ptr();

  std::ostringstream log_stream;
  log_stream << "{" << R"("event": "forceRecordEvent", )" << R"("src_ptr": "0x)"
             << std::hex << (unsigned long long)src_ptr << "\", "
             << "\"src_info\": "
             << at::detail::getCUDAHooks().my_pointerInfo(src_ptr) << ", "
             << R"("dst_ptr": "0x)" << std::hex << (unsigned long long)dst_ptr
             << "\", " << "\"dst_info\": "
             << at::detail::getCUDAHooks().my_pointerInfo(dst_ptr) << ", "
             << R"("recordAtDeviceIdx": ")" << recordAtDeviceIdx << "}\n";
  VLOG(0) << log_stream.str();
  if (self.is_cuda() && self.is_pinned()) {
    VLOG(0) << R"("event" : "forceRecordEventReturn")" << "\n";
    return;
  }
  c10::cuda::CUDAStream stream = detail::getCUDAHooks().current_stream();
  detail::getCUDAHooks().my_recordAndReplaceEvent(
      self, new_tensor, recordAtDeviceIdx.index(), stream);
}

Tensor& my_sync_device_(Tensor& self, Device device) {
  if (self.is_cuda() && self.is_pinned())
    return self;

  c10::cuda::CUDAStream stream = detail::getCUDAHooks().current_stream();
  detail::getCUDAHooks().my_syncEvent(
      self.data_ptr(), device.index(), self.device().index(), stream);
  return self;
}

// tensor is input tensor
Tensor& my_sync_tensor_(Tensor& self, const Tensor& tensor) {
  if (self.is_cuda() && self.is_pinned())
    return self;
  if (!tensor.is_cuda())
    return self;

  std::ostringstream log_stream;
  log_stream << "{" << R"("event": "syncEvent", )" << R"("ptr": "0x)"
             << std::hex << (unsigned long long)self.data_ptr() << "\", "
             << "\"ptr_info\": "
             << at::detail::getCUDAHooks().my_pointerInfo(self.data_ptr())
             << ", " << R"("syncAtDeviceIdx": ")"
             << std::to_string(tensor.device().index()) << "\", "
             << R"("same_device": ")"
             << ((tensor.device().index() == self.device().index()) ? "true"
                                                                    : "false")
             << "\"" << "}\n";
  VLOG(0) << log_stream.str();

  unsigned long long p = (unsigned long long)self.data_ptr();

  c10::cuda::CUDAStream stream = detail::getCUDAHooks().current_stream();
  detail::getCUDAHooks().my_syncEvent(
      self.data_ptr(),
      tensor.device().index(),
      self.device().index(),
      stream,
      false);
  return self;
}

} // namespace native
} // namespace at
