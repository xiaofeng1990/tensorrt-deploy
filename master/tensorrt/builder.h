#ifndef MASTER_TENSORRT_BUILDER_H_
#define MASTER_TENSORRT_BUILDER_H_

#include "common/logging.h"
#include "trt_utils.h"
#include <string>
namespace trt {

enum Mode { FP16 = 0, FP32, INT8 };

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