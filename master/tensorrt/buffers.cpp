#include "buffers.h"
namespace trt {

BufferManager::BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batch_size = 0,
                             const nvinfer1::IExecutionContext *context = nullptr)
    : engine_(engine), batch_size_(batch_size_)
{
}

//!
//! \brief Returns a vector of device buffers that you can use directly as
//!        bindings for the execute and enqueue methods of IExecutionContext.
//!
std::vector<void *> &BufferManager::GetDeviceBindings() {}
const std::vector<void *> &BufferManager::GetDeviceBindings() const {}
//!
//! \brief Returns the device buffer corresponding to tensorName.
//!        Returns nullptr if no such tensor can be found.
//!
void *BufferManager::GetDeviceBuffer(const std::string &tensor_name) const {}
//!
//! \brief Returns the host buffer corresponding to tensorName.
//!        Returns nullptr if no such tensor can be found.
//!
void *BufferManager::GetHostBuffer(const std::string &tensor_name) const {}
//!
//! \brief Returns the size of the host and device buffers that correspond to tensorName.
//!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
//!
size_t BufferManager::Size(const std::string &tensor_name) const {}
//!
//! \brief Dump host buffer with specified tensorName to ostream.
//!        Prints error message to std::ostream if no such tensor can be found.
//!
void BufferManager::DumpBuffer(std::ostream &os, const std::string &tensorName) {}

void BufferManager::CopyInputToDevice() {}
void BufferManager::CopyOutputToHost() {}
//!
//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
//!
void BufferManager::CopyInputToDeviceAsync(const cudaStream_t &stream = 0) {}
//!
//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
//!
void BufferManager::CopyOutputToHostAsync(const cudaStream_t &stream = 0) {}

void *BufferManager::GetBuffer(const bool is_host, const std::string &tensor_name) const {}
void BufferManager::MemcpyBuffers(const bool copy_input, const bool device_to_host, const bool async,
                                  const cudaStream_t &stream = 0)
{
}

} // namespace trt