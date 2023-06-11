#include "buffers.h"
#include "cuda_tools.h"
#include "trt_utils.h"
#include <cassert>
namespace trt {

BufferManager::BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batch_size,
                             const nvinfer1::IExecutionContext *context)
    : engine_(engine), batch_size_(batch_size)
{
    assert(engine->hasImplicitBatchDimension() || batch_size_ == 0);
    // Create host and device buffers input and output
    for (int i = 0; i < engine->getNbBindings(); i++)
    {
        auto dims = context ? context->getBindingDimensions(i) : engine_->getBindingDimensions(i);
        size_t vol = context || !batch_size_ ? 1 : static_cast<size_t>(batch_size_);
        nvinfer1::DataType type = engine_->getBindingDataType(i);
        if (engine_->bindingIsInput(i))
        {
            input_tensort_name_ = engine_->getBindingName(i);
        }
        else
        {
            output_tensort_name_ = engine_->getBindingName(i);
        }

        int vec_dim = engine_->getBindingVectorizedDim(i);
        if (vec_dim != -1)
        {
            int scalars_per_vec = engine_->getBindingComponentsPerElement(i);
            dims.d[vec_dim] = div_up(dims.d[vec_dim], scalars_per_vec);
            vol *= scalars_per_vec;
        }
        vol *= volume(dims);
        std::unique_ptr<ManagedBuffer> man_buf{new ManagedBuffer()};
        man_buf->device_buffer_ = DeviceBuffer(vol, type);
        man_buf->host_buffer_ = HostBuffer(vol, type);
        // bind device buffer: input GPU buffer and output GPU buffer
        device_bindings_.emplace_back(man_buf->device_buffer_.data());
        managed_buffers_.emplace_back(std::move(man_buf));
    }
}

//!
//! \brief Returns a vector of device buffers that you can use directly as
//!        bindings for the execute and enqueue methods of IExecutionContext.
//!
std::vector<void *> &BufferManager::GetDeviceBindings() { return device_bindings_; }
const std::vector<void *> &BufferManager::GetDeviceBindings() const { return device_bindings_; }
//!
//! \brief Returns the device buffer corresponding to tensorName.
//!        Returns nullptr if no such tensor can be found.
//!
void *BufferManager::GetDeviceBuffer(const std::string &tensor_name) const { return GetBuffer(false, tensor_name); }
void *BufferManager::GetDeviceInputBuffer() const { return GetBuffer(false, input_tensort_name_); }
void *BufferManager::GetDeviceOutputBuffer() const { return GetBuffer(false, output_tensort_name_); }

//!
//! \brief Returns the host buffer corresponding to tensorName.
//!        Returns nullptr if no such tensor can be found.
//!
void *BufferManager::GetHostBuffer(const std::string &tensor_name) const { return GetBuffer(true, tensor_name); }
void *BufferManager::GetHostInputBuffer() const { return GetBuffer(true, input_tensort_name_); }
void *BufferManager::GetHostOutputBuffer() const { return GetBuffer(true, output_tensort_name_); }
//!
//! \brief Returns the size of the host and device buffers that correspond to tensorName.
//!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
//!
size_t BufferManager::Size(const std::string &tensor_name) const
{
    int index = engine_->getBindingIndex(tensor_name.c_str());
    if (index == -1)
        return kINVALID_SIZE_VALUE;
    return managed_buffers_[index]->host_buffer_.nb_bytes();
}
//!
//! \brief Dump host buffer with specified tensorName to ostream.
//!        Prints error message to std::ostream if no such tensor can be found.
//!
void BufferManager::DumpBuffer(std::ostream &os, const std::string &tensorName)
{
    // int index = engine_->getBindingIndex(tensorName.c_str());
    // if (index == -1)
    // {
    //     os << "Invalid tensor name" << std::endl;
    //     return;
    // }
    // void*buf = managed_buffers_[index]->host_buffer_.data();
    // size_t  buf_size =managed_buffers_[index]->host_buffer_.nb_bytes();
    // nvinfer1::Dims buf_dims = engine_->getBindingDimensions(index);
}

void BufferManager::CopyInputToDevice() { MemcpyBuffers(true, false, false); }
void BufferManager::CopyOutputToHost() { MemcpyBuffers(false, true, false); }
//!
//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
//!
void BufferManager::CopyInputToDeviceAsync(const cudaStream_t &stream) { MemcpyBuffers(true, false, true, stream); }
//!
//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
//!
void BufferManager::CopyOutputToHostAsync(const cudaStream_t &stream) { MemcpyBuffers(false, true, true, stream); }

void *BufferManager::GetBuffer(const bool is_host, const std::string &tensor_name) const
{
    int index = engine_->getBindingIndex(tensor_name.c_str());
    if (index == -1)
        return nullptr;
    return (is_host ? managed_buffers_[index]->host_buffer_.data() : managed_buffers_[index]->device_buffer_.data());
}
void BufferManager::MemcpyBuffers(const bool copy_input, const bool device_to_host, const bool async,
                                  const cudaStream_t &stream)
{
    for (int i = 0; i < engine_->getNbBindings(); i++)
    {
        void *dst_ptr =
            device_to_host ? managed_buffers_[i]->host_buffer_.data() : managed_buffers_[i]->device_buffer_.data();
        void *src_ptr =
            device_to_host ? managed_buffers_[i]->device_buffer_.data() : managed_buffers_[i]->host_buffer_.data();
        const size_t byte_size = managed_buffers_[i]->host_buffer_.nb_bytes();
        const cudaMemcpyKind memcpy_type = device_to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        if ((copy_input && engine_->bindingIsInput(i)) || (!copy_input && !engine_->bindingIsInput(i)))
        {
            if (async)
            {
                checkRuntime(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, memcpy_type, stream));
            }
            else
            {
                checkRuntime(cudaMemcpy(dst_ptr, src_ptr, byte_size, memcpy_type));
            }
        }
    }
}

std::string BufferManager::GetInputTensorName() { return input_tensort_name_; }
std::string BufferManager::GetOutputTensorName() { return output_tensort_name_; }
void BufferManager::SetDeviceInputBuffer() {}
} // namespace trt