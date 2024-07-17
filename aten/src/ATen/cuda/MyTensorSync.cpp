#include <ATen/core/Tensor.h>
#include <ATen/cuda/MyTensorSync.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Logging.h>
#include <driver_types.h>
#include <fmt/format.h>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

namespace at {

std::string pointerInfo(TensorId ptr) {
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr));

  std::ostringstream json_info;

  cudaMemoryType memoryType = attributes.type;
  json_info << "{";

  switch (memoryType) {
    case cudaMemoryTypeHost:
      json_info << R"("memoryType": "Host memory")";
      break;
    case cudaMemoryTypeDevice:
      json_info << R"("memoryType": "Device memory", "device": )"
                << attributes.device;
      break;
    case cudaMemoryTypeManaged:
      json_info << R"("memoryType": "Managed memory")";
      break;
    default:
      json_info << R"("memoryType": "Unknown memory type")";
      break;
  }

  json_info << "}";
  return json_info.str();
}

class DeviceEventManager {
  std::string recordEventLog(
      const std::string& eventName,
      TensorId srcId,
      TensorId dstId,
      c10::DeviceIndex deviceIdx,
      cudaEvent_t cudaEvent) {
    return fmt::format(
        R"({{"event": "{}", "src_ptr": "{}", "src_info": {}, "dst_ptr": "{}", "dst_info": {}, "recordAtDeviceIdx": "{}","cudaEvent": "{}"}})",
        eventName,
        fmt::ptr(srcId),
        pointerInfo(srcId),
        fmt::ptr(dstId),
        pointerInfo(dstId),
        deviceIdx,
        fmt::ptr(cudaEvent));
  }

  std::string syncEventLog(
      const std::string& eventName,
      TensorId tensorId,
      c10::DeviceIndex deviceIdx,
      bool same_device,
      bool isFindEvent) {
    return fmt::format(
        R"({{"event": "{}", "ptr": "{}", "ptr_info": {}, "syncAtDeviceIdx": "{}", "same_device": "{}", "findEvent": "{}"}})",
        eventName,
        fmt::ptr(tensorId),
        pointerInfo(tensorId),
        deviceIdx,
        same_device ? "true" : "false",
        isFindEvent ? "true" : "false");
  }

 public:
  DeviceEventManager(c10::DeviceIndex deviceIdx) : deviceIdx(deviceIdx){};

  void recordAndReplaceEvent(
      at::Tensor& src,
      const at::Tensor& dst,
      at::cuda::CUDAStream stream) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      TensorId srcId = src.data_ptr();
      TensorId dstId;

      src.set_data(dst);
      assert(src.is_cuda() && "src must be a CUDA tensor");
      dstId = src.data_ptr();

      cudaEvent_t event = createCudaEvent();
      C10_CUDA_CHECK(cudaEventRecord(event, stream)); // 传输完成事件

      VLOG(0) << recordEventLog(
          "recordAndReplaceEvent", srcId, dstId, deviceIdx, event);

      // Deletes items that already exist.
      auto it = eventMap.find(srcId);
      eventMap[srcId] = {dstId, event};

      it = eventMap.find(dstId);
      eventMap[dstId] = {srcId, event};
    }
    cv.notify_all();
  }

  void recordEvent(
      TensorId srcId,
      TensorId dstId,
      at::cuda::CUDAStream stream) {
    {
      std::lock_guard<std::mutex> lock(mutex);

      // assert(src.is_cuda() && "src must be a CUDA tensor");

      cudaEvent_t event = createCudaEvent();
      C10_CUDA_CHECK(cudaEventRecord(event, stream)); // 传输完成事件

      VLOG(0) << recordEventLog("recordEvent", srcId, dstId, deviceIdx, event);

      // Deletes items that already exist.
      auto it = eventMap.find(srcId);
      eventMap[srcId] = {dstId, event};

      it = eventMap.find(dstId);
      eventMap[dstId] = {srcId, event};
    }
    cv.notify_all();
  }

  void syncEvent(
      TensorId id,
      at::cuda::CUDAStream stream,
      bool same_device,
      bool enableLog = true) {
    cudaEvent_t event;
    {
      std::unique_lock<std::mutex> lock(mutex);
      auto it = eventMap.find(id);

      if (enableLog) {
        VLOG(0) << syncEventLog(
            "syncEvent", id, deviceIdx, same_device, it != eventMap.end());
      }

      if (it == eventMap.end()) {
        if (same_device) {
          return;
        }
        cv.wait(lock, [&] {
          VLOG(0) << syncEventLog(
              "syncEventWait",
              id,
              deviceIdx,
              same_device,
              it != eventMap.end());
          auto _it = this->eventMap.find(id);
          return _it != this->eventMap.end();
        });
        it = eventMap.find(id);
      }

      event = it->second.second;
      TensorId linked_ptr = it->second.first;
      eventMap.erase(it);
      it = eventMap.find(linked_ptr);
      if (it != eventMap.end()) {
        eventMap.erase(it);
      }
    }

    // C10_CUDA_CHECK(cudaEventSynchronize(event));
    // freeCudaEvent(event);

    auto log = fmt::format(
        R"({{"event": "{}", "ptr": "{}", "ptr_info": {}, "syncAtDeviceIdx":
        "{}", "cudaEvent": "{}"}})",
        "syncEventStreamWaitEvent",
        fmt::ptr(id),
        pointerInfo(id),
        deviceIdx,
        fmt::ptr(event));
    VLOG(0) << log;

    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
  }

  cudaEvent_t createCudaEvent() {
    cudaEvent_t event = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  void freeCudaEvent(cudaEvent_t event) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }

  std::mutex mutex;
  std::condition_variable cv;
  c10::DeviceIndex deviceIdx;
  std::map<TensorId, std::pair<TensorId, cudaEvent_t>> eventMap; // 记录交换事件

  std::map<TensorId, std::pair<TensorId, cudaEvent_t>>
      tensorChunkEventMap; // 记录分割事件
};

std::vector<std::unique_ptr<DeviceEventManager>>& getSingletonVector() {
  static std::vector<std::unique_ptr<DeviceEventManager>> instance;
  return instance;
}

void initDeviceEventManager(int deviceCount) {
  auto& inst = getSingletonVector();
  static bool initialized = false;
  if (!initialized) {
    auto size = inst.size();
    if (size < deviceCount) {
      inst.resize(deviceCount);
      for (int i = 0; i < deviceCount; i++) {
        inst[i] = std::make_unique<DeviceEventManager>(i);
      }
    }
    initialized = true;
  }
}

void recordAndReplaceEvent(
    at::Tensor& src,
    const at::Tensor& new_tensor,
    c10::DeviceIndex recordAtDeviceIdx,
    at::cuda::CUDAStream stream) {
  getSingletonVector()[recordAtDeviceIdx]->recordAndReplaceEvent(
      src, new_tensor, stream);
}

void syncEvent(
    TensorId ptr,
    c10::DeviceIndex syncAtDeviceIdx,
    c10::DeviceIndex currentTensorDeviceIdx,
    at::cuda::CUDAStream stream,
    bool enableLog) {
  getSingletonVector()[syncAtDeviceIdx]->syncEvent(
      ptr, stream, syncAtDeviceIdx == currentTensorDeviceIdx, enableLog);
}

void recordEvent(
    TensorId srcId,
    TensorId dstId,
    c10::DeviceIndex recordAtDeviceIdx,
    at::cuda::CUDAStream stream) {
  getSingletonVector()[recordAtDeviceIdx]->recordEvent(srcId, dstId, stream);
}

void clear() {
  for (auto& deviceEventManager : getSingletonVector()) {
    deviceEventManager->eventMap.clear();
    deviceEventManager->tensorChunkEventMap.clear();
  }
}

} // namespace at
