#ifndef MASTER_TENSORRT_TENSOR_BUFFERS_H_
#define MASTER_TENSORRT_TENSOR_BUFFERS_H_
#include "generic_buffer.h"
#include "trt_utils.h"
#include <NvInfer.h>
#include <memory>
#include <vector>

namespace trt {

enum DataHandle { Init = 0, Device = 1, Host = 2 };

//!
//! \brief  The Buffer class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class TensorBufferManager {

  public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
    TensorBufferManager(size_t size, int type_size);
    ~TensorBufferManager() = default;

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void *GetDeviceBuffer() const;

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void *GetHostBuffer() const;

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t Size() const;
    //!
    //! \brief Dump host buffer with specified tensorName to ostream.
    //!        Prints error message to std::ostream if no such tensor can be found.
    //!
    void DumpBuffer(std::ostream &os);
    template <typename T>
    void Print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount);
    void CopyToDevice();
    void CopyToHost();
    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void CopyToDeviceAsync(const cudaStream_t &stream = 0);
    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void CopyToHostAsync(const cudaStream_t &stream = 0);

  private:
    void *GetBuffer(const bool is_host) const;
    void MemcpyBuffers(const bool device_to_host, const bool async, const cudaStream_t &stream = 0);

  private:
    //!< The vector of pointers to managed buffers， input and output
    std::unique_ptr<ManagedBuffer> managed_buffers_;
    DataHandle handle_{Init};
    size_t size_;
};

template <typename T>
void TensorBufferManager::Print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount)
{
}

} // namespace trt

#endif // MASTER_TENSORRT_BUFFERS_H_