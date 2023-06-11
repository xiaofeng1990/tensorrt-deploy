#ifndef MASTER_TENSORRT_BUFFERS_H_
#define MASTER_TENSORRT_BUFFERS_H_
#include "trt_utils.h"
#include <NvInfer.h>
#include <memory>
#include <vector>
namespace trt {
//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
  public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : size_(0), capacity_(0), type_(type), buffer_(nullptr)
    {
    }
    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size, nvinfer1::DataType type) : size_(size), capacity_(size), type_(type)
    {
        if (!alloc_fn_(&buffer_, this->nb_bytes()))
        {
            throw std::bad_alloc();
        }
    }
    // move
    GenericBuffer(GenericBuffer &&buf)
        : size_(buf.size_), capacity_(buf.capacity_), type_(buf.type_), buffer_(buf.buffer_)
    {
        buf.size_ = 0;
        buf.capacity_ = 0;
        buf.type_ = nvinfer1::DataType::kFLOAT;
        buf.buffer_ = nullptr;
    }
    GenericBuffer &operator=(GenericBuffer &&buf)
    {
        if (this != &buf)
        {
            free_fn_(buffer_);
            size_ = buf.size_;
            capacity_ = buf.capacity_;
            type_ = buf.type_;
            buffer_ = buf.buffer_;
            buf.size_ = 0;
            buf.capacity_ = 0;
            buf.buffer_ = nullptr;
        }
        return *this;
    }

    ~GenericBuffer() { free_fn_(buffer_); }
    // returns pointer to underlying array
    void *data() { return buffer_; }
    // returns pointer to underlying array
    const void *data() const { return buffer_; }
    // Returns the size (in number of elements) of the buffer.
    size_t size() const { return size_; }
    // Returns the size(in bytes) of the buffer.
    size_t nb_bytes() const { return this->size() * get_element_size(type_); }
    // Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    void resize(size_t new_size)
    {
        size_ = new_size;
        if (capacity_ < new_size)
        {
            free_fn_(buffer_);
            if (alloc_fn_(&buffer_, this->nb_bytes()))
            {
                throw std::bad_alloc();
            }
            capacity_ = new_size;
        }
    }

    void resize(const nvinfer1::Dims &dims) { return this->resize(trt::volume(dims)); }

  private:
    size_t size_{0};
    size_t capacity_{0};
    nvinfer1::DataType type_;
    void *buffer_;
    AllocFunc alloc_fn_;
    FreeFunc free_fn_;
};

class DeviceAllocator {
  public:
    bool operator()(void **ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};
class DeviceFree {
  public:
    void operator()(void *ptr) const { cudaFree(ptr); }
};

class HostAllocator {
  public:
    bool operator()(void **ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree {
  public:
    void operator()(void *ptr) const { free(ptr); }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

class ManagedBuffer {
  public:
    DeviceBuffer device_buffer_;
    HostBuffer host_buffer_;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager {

  public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batch_size = 0,
                  const nvinfer1::IExecutionContext *context = nullptr);
    ~BufferManager() = default;
    //!
    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void *> &GetDeviceBindings();
    const std::vector<void *> &GetDeviceBindings() const;
    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void *GetDeviceBuffer(const std::string &tensor_name) const;
    void *GetDeviceInputBuffer() const;
    void *GetDeviceOutputBuffer() const;
    void SetDeviceInputBuffer();
    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void *GetHostBuffer(const std::string &tensor_name) const;
    void *GetHostInputBuffer() const;
    void *GetHostOutputBuffer() const;

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t Size(const std::string &tensor_name) const;
    //!
    //! \brief Dump host buffer with specified tensorName to ostream.
    //!        Prints error message to std::ostream if no such tensor can be found.
    //!
    void DumpBuffer(std::ostream &os, const std::string &tensorName);
    template <typename T>
    void Print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount);
    void CopyInputToDevice();
    void CopyOutputToHost();
    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void CopyInputToDeviceAsync(const cudaStream_t &stream = 0);
    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void CopyOutputToHostAsync(const cudaStream_t &stream = 0);
    std::string GetInputTensorName();
    std::string GetOutputTensorName();

  private:
    void *GetBuffer(const bool is_host, const std::string &tensor_name) const;
    void MemcpyBuffers(const bool copy_input, const bool device_to_host, const bool async,
                       const cudaStream_t &stream = 0);

  private:
    //!< The pointer to the engine
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    //!< The batch size for legacy networks, 0 otherwise.
    int batch_size_;
    //!< The vector of pointers to managed buffers， input and output
    std::vector<std::unique_ptr<ManagedBuffer>> managed_buffers_;
    //!< The vector of device buffers needed for engine execution
    std::vector<void *> device_bindings_;
    std::string input_tensort_name_;
    std::string output_tensort_name_;
};

template <typename T>
void BufferManager::Print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount)
{
}

} // namespace trt

#endif // MASTER_TENSORRT_BUFFERS_H_