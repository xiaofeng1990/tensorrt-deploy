#include "tensor_buffer.h"
#include "cuda_tools.h"
#include <cassert>
namespace trt {

TensorBufferManager::TensorBufferManager(size_t size, int type_size) : size_(size)
{

    assert(size == 0);
    managed_buffers_.reset(new ManagedBuffer);
    managed_buffers_->host_buffer_ = HostBuffer(size, type_size);
    managed_buffers_->device_buffer_ = DeviceBuffer(size, type_size);

    handle_ = Init;
}

//!
//! \brief Returns the device buffer.
//!        Returns nullptr if no such tensor can be found.
//!
void *TensorBufferManager::GetDeviceBuffer() const { return GetBuffer(false); }

//!
//! \brief Returns the host buffer.
//!        Returns nullptr if no such tensor can be found.
//!
void *TensorBufferManager::GetHostBuffer() const { return GetBuffer(true); }

//!
//! \brief Returns the size of the host and device buffers that correspond to tensorName.
//!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
//!
size_t TensorBufferManager::Size() const { return managed_buffers_->host_buffer_.nb_bytes(); }
//!
//! \brief Dump host buffer with specified tensorName to ostream.
//!        Prints error message to std::ostream if no such tensor can be found.
//!
void TensorBufferManager::DumpBuffer(std::ostream &os) {}

void TensorBufferManager::CopyToDevice() { MemcpyBuffers(false, false); }
void TensorBufferManager::CopyToHost() { MemcpyBuffers(true, false); }
//!
//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
//!
void TensorBufferManager::CopyToDeviceAsync(const cudaStream_t &stream = 0) { MemcpyBuffers(false, true, stream); }
//!
//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
//!
void TensorBufferManager::CopyToHostAsync(const cudaStream_t &stream = 0) { MemcpyBuffers(true, true, stream); }

void *TensorBufferManager::GetBuffer(const bool is_host) const
{
    return is_host ? managed_buffers_->host_buffer_.data() : managed_buffers_->device_buffer_.data();
}
void TensorBufferManager::MemcpyBuffers(const bool device_to_host, const bool async, const cudaStream_t &stream = 0)
{
    void *dst_ptr = device_to_host ? managed_buffers_->host_buffer_.data() : managed_buffers_->device_buffer_.data();
    void *src_ptr = device_to_host ? managed_buffers_->device_buffer_.data() : managed_buffers_->host_buffer_.data();
    const size_t byte_size = managed_buffers_->host_buffer_.nb_bytes();
    const cudaMemcpyKind memcpy_type = device_to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    if (async)
    {
        checkRuntime(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, memcpy_type, stream));
    }
    else
    {
        checkRuntime(cudaMemcpy(dst_ptr, src_ptr, byte_size, memcpy_type));
    }
}
} // namespace trt
