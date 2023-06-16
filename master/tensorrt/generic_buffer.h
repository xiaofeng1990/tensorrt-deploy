
#ifndef MASTER_TENSORRT_GENERIC_BUFFER_H_
#define MASTER_TENSORRT_GENERIC_BUFFER_H_
#include "common/logging.h"
#include "trt_utils.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
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
        type_size_ = get_element_size(type_);
    }
    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size, nvinfer1::DataType type) : size_(size), capacity_(size), type_(type)
    {
        type_size_ = get_element_size(type_);
        if (!alloc_fn_(&buffer_, this->nb_bytes()))
        {
            XF_LOGT(ERROR, TAG, "alloc memory by data type failed");
            throw std::bad_alloc();
        }
    }

    GenericBuffer(size_t size, int type_size) : size_(size), capacity_(size), type_size_(type_size)
    {
        if (!alloc_fn_(&buffer_, this->nb_bytes()))
        {
            XF_LOGT(ERROR, TAG, "alloc memory by data type size failed");
            throw std::bad_alloc();
        }
    }

    // move
    GenericBuffer(GenericBuffer &&buf)
        : size_(buf.size_), capacity_(buf.capacity_), type_(buf.type_), buffer_(buf.buffer_), type_size_(buf.type_size_)
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
            type_size_ = buf.type_size_;
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
    size_t nb_bytes() const { return this->size() * type_size_; }
    // Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    void resize(size_t new_size)
    {
        size_ = new_size;
        if (capacity_ < new_size)
        {
            free_fn_(buffer_);
            if (alloc_fn_(&buffer_, this->nb_bytes()))
            {
                XF_LOGT(ERROR, TAG, "resize memory failed");
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
    int type_size_;
    void *buffer_;
    AllocFunc alloc_fn_;
    FreeFunc free_fn_;
    static constexpr const char *TAG = "generic_buffer";
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
} // namespace trt

#endif // MASTER_TENSORRT_GENERIC_BUFFER_H_
