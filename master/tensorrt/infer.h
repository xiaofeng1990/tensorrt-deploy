#ifndef MASTER_TENSORRT_INFER_H_
#define MASTER_TENSORRT_INFER_H_

#include "engine_buffers.h"
#include "trt_logger.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace trt {
class Infer {
  public:
    Infer() = default;
    ~Infer();
    //单个图像推理
    void *Inference(const std::vector<cv::Mat> &images);
    void *Inference(const cv::Mat &images);
    void *InferenceAsync(std::vector<cv::Mat> images);
    // batch 推理

    //接在engine文件到context

    bool LoadModel(std::string engine_file);
    //获取模型属性  input_name output_name batch
    void Destroy();
    //同步stream
    void SynchronizeStream();
    int GetMaxBatchSize();
    void GetModelProperty();
    std::vector<int> GetOutputShape();
    std::vector<int> GetInputShape();
    void TestModel();
    void TestModel2();

  private:
    std::vector<unsigned char> LoadEngine(std::string engine_file);

  private:
    bool model_status_ = false;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::shared_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    std::shared_ptr<nvinfer1::IExecutionContext> context_{nullptr};
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<EngineBufferManager> buffers_{nullptr};
    // input output name shape
    std::string input_tensort_name_;
    std::string output_tensort_name_;
    int max_batch_size_;
    int opt_batch_size_;
    int min_batch_size_;
    int output_buffer_size_;
    int input_buffer_size_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    static constexpr const char *TAG = "infer";
    TRTLogger gLogger;
};
} // namespace trt

#endif // MASTER_TENSORRT_INFER_H_
