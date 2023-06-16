#include "infer.h"
#include "common/logging.h"
#include "cuda_tools.h"
#include "trt_utils.h"
#include <fstream>
#include <iostream>

namespace trt

{

Infer::~Infer() { Destroy(); }
//同步图像推理
bool Infer::Inference(std::vector<cv::Mat> images)
{
    // Memcpy from host input buffers to device input buffers
    // set mem to input host
    // copy host mem to device mem
    buffers_->CopyInputToDevice();

    // Synchronously execute inference a network.
    bool status = context_->executeV2(buffers_->GetDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers_->CopyOutputToHost();
    // return output host mem
    return true;
}

//异步推理
bool Infer::InferenceAsync(std::vector<cv::Mat> images)
{
    // Memcpy from host input buffers to device input buffers
    // set mem to input host
    // copy host mem to device mem
    buffers_->CopyInputToDeviceAsync(stream_);
    //异步推理
    bool success = context_->enqueueV2(buffers_->GetDeviceBindings().data(), stream_, nullptr);
    // Memcpy from device output buffers to host output buffers
    buffers_->CopyOutputToHostAsync(stream_);
    SynchronizeStream();

    // return output host mem
    return true;
}
// batch 推理

std::vector<unsigned char> Infer::LoadEngine(std::string engine_file)
{
    std::ifstream in(engine_file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();
    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char *)&data[0], length);
    }
    in.close();

    return data;
}

//接在engine文件到context
bool Infer::LoadModel(std::string engine_file)
{
    //加载engine file
    auto engine_data = LoadEngine(engine_file);
    runtime_ = make_shared_ptr(nvinfer1::createInferRuntime(gLogger));
    //序列化engine
    engine_ = make_shared_ptr(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine_ == nullptr)
    {
        XF_LOGT(ERROR, TAG, "Deserialize cuda engine failed!");
        runtime_->destroy();
        return false;
    }
    //
    checkRuntime(cudaStreamCreate(&stream_));
    if (stream_ == nullptr)
    {
        XF_LOGT(ERROR, TAG, "Create stream failed!");
        return false;
    }

    context_ = make_shared_ptr(engine_->createExecutionContext());
    if (context_ == nullptr)
    {
        XF_LOGT(ERROR, TAG, "Create context failed!");
        return false;
    }
    max_batch_size_ = engine_->getMaxBatchSize();
    buffers_.reset(new EngineBufferManager(engine_, max_batch_size_));
    input_tensort_name_ = buffers_->GetInputTensorName();
    output_tensort_name_ = buffers_->GetOutputTensorName();
    model_status_ = true;
    return true;
}

//获取模型属性  input_name output_name batch

//同步stream
void Infer::SynchronizeStream() { checkRuntime(cudaStreamSynchronize(stream_)); }

int Infer::GetMaxBatchSize() { return max_batch_size_; }

void Infer::Destroy()
{
    // context_.reset();
    // engine_.reset();
    // runtime_.reset();
    // buffers_.reset();
    cudaStreamDestroy(stream_);
}
} // namespace trt
