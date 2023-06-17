#ifndef MASTER_TENSORRT_ENGINE_BUFFERS_H_
#define MASTER_TENSORRT_ENGINE_BUFFERS_H_
#include "generic_buffer.h"
#include "trt_utils.h"
#include <NvInfer.h>
#include <memory>
#include <vector>

namespace trt {

//!
//! \brief  The EngineBufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The EngineBufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class EngineBufferManager {

  public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
    EngineBufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batch_size = 0);
    ~EngineBufferManager() = default;
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
void EngineBufferManager::Print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount)
{
}

} // namespace trt

#endif // MASTER_TENSORRT_BUFFERS_H_