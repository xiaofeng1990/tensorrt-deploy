#ifndef MASTER_TENSORRT_BUILDER_H_
#define MASTER_TENSORRT_BUILDER_H_

#include "common/logging.h"
#include "trt_utils.h"
#include <string>
namespace trt {

enum Mode { FP16 = 0, FP32, INT8 };

class Logger : public nvinfer1::ILogger {
  public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            XF_LOGT(ERROR, TAG, "NVInfer INTERNAL_ERROR: %s", msg);
            abort();
            break;
        case Severity::kERROR:
            XF_LOGT(ERROR, TAG, "NVInfer: %s", msg);
            break;
        case Severity::kWARNING:
            XF_LOGT(WARN, TAG, "NVInfer: %s", msg);
            break;
        // case Severity::kINFO:
        //     XF_LOGT(INFO, TAG, "NVInfer: %s", msg);
        // case Severity::kVERBOSE:
        //     XF_LOGT(DEBUG, TAG, "NVInfer: %s", msg);
        //     break;
        default:
            break;
        }
    }

  private:
    static constexpr const char *TAG = "tlt_loger";
};

class Builder {

  public:
    Builder(/* args */);
    ~Builder();
    bool Compile(Mode mode, unsigned int max_batch_size, const std::string onnx_file, const std::string engine_file,
                 const size_t max_workspace_size = 28_MiB);

    const char *ModeTosString(Mode type);
    bool SaveEngineFile(const std::string engine_file, const void *data, size_t length);

  private:
    static constexpr const char *TAG = "builder";
}; // namespace trt

} // namespace trt

#endif