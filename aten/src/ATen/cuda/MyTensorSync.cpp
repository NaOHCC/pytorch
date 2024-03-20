#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime_api.h>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>

struct DeviceEventManager {
 public:
  DeviceEventManager(int device) : device(device){};

  void recordEvent(
      at::Tensor& src,
      const at::Tensor& dst,
      at::cuda::CUDAStream stream) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      void* src_ptr = src.data_ptr();
      void* dst_ptr;

      src.set_data(dst);

      dst_ptr = src.data_ptr();

      cudaEvent_t event = createCudaEvent();
      C10_CUDA_CHECK(cudaEventRecord(event, stream));

      // Deletes items that already exist.
      auto it = recorded_event_map.find(src_ptr);
      recorded_event_map[src_ptr] = {dst_ptr, event};

      it = recorded_event_map.find(dst_ptr);
      recorded_event_map[dst_ptr] = {src_ptr, event};
    }
    cv.notify_all();
  }

  void syncEvent(void* ptr, at::cuda::CUDAStream stream, bool same_device) {
    cudaEvent_t event;
    {
      std::unique_lock<std::mutex> lock(mutex);
      auto it = recorded_event_map.find(ptr);
      if (it == recorded_event_map.end()) {
        if (same_device) {
          return;
        }
        cv.wait(lock, [&] {
          auto _it = this->recorded_event_map.find(ptr);
          return _it != this->recorded_event_map.end();
        });
        it = recorded_event_map.find(ptr);
      }

      event = it->second.second;
      void* linked_ptr = it->second.first;
      recorded_event_map.erase(it);
      it = recorded_event_map.find(linked_ptr);
      if (it != recorded_event_map.end()) {
        recorded_event_map.erase(it);
      }
    }

    // C10_CUDA_CHECK(cudaEventSynchronize(event));
    // freeCudaEvent(event);
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
  }

  cudaEvent_t createCudaEvent() {
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return event;
  }

  void freeCudaEvent(cudaEvent_t event) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }

  std::mutex mutex;
  std::condition_variable cv;
  int device;
  std::map<void*, std::pair<void*, cudaEvent_t>> recorded_event_map;
};

struct EventManager {
 public:
  void init(int device_count) {
    int size = device_event_managers.size();
    if (size < device_count) {
      device_event_managers.resize(device_count);
      for (int i = 0; i < device_count; i++) {
        device_event_managers[i] =
            std::unique_ptr<DeviceEventManager>(new DeviceEventManager(i));
      }
    }
  }

  void recordEvent(
      at::Tensor& src,
      const at::Tensor& dst,
      int device,
      at::cuda::CUDAStream stream) {
    device_event_managers[device]->recordEvent(src, dst, stream);
  }

  void syncEvent(
      void* ptr,
      int device,
      int cur_device,
      at::cuda::CUDAStream stream) {
    device_event_managers[device]->syncEvent(ptr, stream, device == cur_device);
  }

  std::vector<std::unique_ptr<DeviceEventManager>> device_event_managers;
};

static EventManager event_manager;

void initEventManager(int device_count) {
  event_manager.init(device_count);
}

void recordEvent(
    at::Tensor& self,
    const at::Tensor& tensor,
    int device,
    at::cuda::CUDAStream stream) {
  event_manager.recordEvent(self, tensor, device, stream);
}

void syncEvent(
    void* ptr,
    int device,
    int cur_device,
    at::cuda::CUDAStream stream) {
  event_manager.syncEvent(ptr, device, cur_device, stream);
}
