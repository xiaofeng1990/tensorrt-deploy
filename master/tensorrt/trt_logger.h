#ifndef MASTER_TENSORRT_TRT_LOGGER_H_
#define MASTER_TENSORRT_TRT_LOGGER_H_
#include "common/logging.h"
#include <NvInfer.h>
namespace trt {

class TRTLogger : public nvinfer1::ILogger {
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
} // namespace trt
#endif // MASTER_TENSORRT_TRT_LOGGER_H_