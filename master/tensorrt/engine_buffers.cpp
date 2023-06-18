#include "engine_buffers.h"
#include "cuda_tools.h"
#include "trt_utils.h"
#include <cassert>
namespace trt {

EngineBufferManager::EngineBufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batch_size)
    : engine_(engine), batch_size_(batch_size)
{
    // assert(engine->hasImplicitBatchDimension() || batch_size_ == 0);
    // Create host and device buffers input and output

    for (int i = 0; i < engine->getNbBindings(); i++)
    {
        auto dims = engine_->getBindingDimensions(i);
        size_t vol = 1;
        if (dims.d[0] <= 0)
        {
            dims.d[0] = static_cast<size_t>(batch_size_);
        }

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
// BufferManager::~BufferManager()
// {
//     if (engine_)
//         engine_.reset();
//     managed_buffers_.clear();
// }
//!
//! \brief Returns a vector of device buffers that you can use directly as
//!        bindings for the execute and enqueue methods of IExecutionContext.
//!
std::vector<void *> &EngineBufferManager::GetDeviceBindings() { return device_bindings_; }
const std::vector<void *> &EngineBufferManager::GetDeviceBindings() const { return device_bindings_; }
//!
//! \brief Returns the device buffer corresponding to tensorName.
//!        Returns nullptr if no such tensor can be found.
//!
void *EngineBufferManager::GetDeviceBuffer(const std::string &tensor_name) const
{
    return GetBuffer(false, tensor_name);
}
void *EngineBufferManager::GetDeviceInputBuffer() const { return GetBuffer(false, input_tensort_name_); }
void *EngineBufferManager::GetDeviceOutputBuffer() const { return GetBuffer(false, output_tensort_name_); }

//!
//! \brief Returns the host buffer corresponding to tensorName.
//!        Returns nullptr if no such tensor can be found.
//!
void *EngineBufferManager::GetHostBuffer(const std::string &tensor_name) const { return GetBuffer(true, tensor_name); }
void *EngineBufferManager::GetHostInputBuffer() const { return GetBuffer(true, input_tensort_name_); }
void *EngineBufferManager::GetHostOutputBuffer() const { return GetBuffer(true, output_tensort_name_); }
//!
//! \brief Returns the size of the host and device buffers that correspond to tensorName.
//!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
//!
size_t EngineBufferManager::Size(const std::string &tensor_name) const
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
void EngineBufferManager::DumpBuffer(std::ostream &os, const std::string &tensorName)
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

void EngineBufferManager::CopyInputToDevice() { MemcpyBuffers(true, false, false); }
void EngineBufferManager::CopyOutputToHost() { MemcpyBuffers(false, true, false); }
//!
//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
//!
void EngineBufferManager::CopyInputToDeviceAsync(const cudaStream_t &stream)
{
    MemcpyBuffers(true, false, true, stream);
}
//!
//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
//!
void EngineBufferManager::CopyOutputToHostAsync(const cudaStream_t &stream)
{
    MemcpyBuffers(false, true, true, stream);
}

void *EngineBufferManager::GetBuffer(const bool is_host, const std::string &tensor_name) const
{
    int index = engine_->getBindingIndex(tensor_name.c_str());
    if (index == -1)
        return nullptr;
    return (is_host ? managed_buffers_[index]->host_buffer_.data() : managed_buffers_[index]->device_buffer_.data());
}
void EngineBufferManager::MemcpyBuffers(const bool copy_input, const bool device_to_host, const bool async,
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

std::string EngineBufferManager::GetInputTensorName() { return input_tensort_name_; }
std::string EngineBufferManager::GetOutputTensorName() { return output_tensort_name_; }
void EngineBufferManager::SetDeviceInputBuffer() {}

void EngineBufferManager::SetHostInputBuffer(const std::vector<cv::Mat> &images)
{
    float *host_input_buffer = static_cast<float *>(GetHostInputBuffer());
    for (auto image : images)
    {
        int image_area = image.cols * image.rows;
        unsigned char *pimage = image.data;
        float *phost_b = host_input_buffer + image_area * 0;
        float *phost_g = host_input_buffer + image_area * 1;
        float *phost_r = host_input_buffer + image_area * 2;
        for (int i = 0; i < image_area; ++i, pimage += 3)
        {
            // 注意这里的顺序rgb调换了
            *phost_r++ = pimage[0] / 255.0f;
            *phost_g++ = pimage[1] / 255.0f;
            *phost_b++ = pimage[2] / 255.0f;
        }
        size_t size_image = image.cols * image.rows * 3;
        host_input_buffer += size_image;
    }
}

} // namespace trt