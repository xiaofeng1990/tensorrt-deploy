#include "infer.h"
#include "builder.h"
#include "common/logging.h"
#include "cuda_tools.h"
#include "trt_utils.h"
#include <fstream>
#include <iostream>
namespace trt

{

Infer::~Infer() { Destroy(); }
//同步图像推理
void *Infer::Inference(const std::vector<cv::Mat> &images)
{
    // Memcpy from host input buffers to device input buffers
    // set mem to input host
    // copy host mem to device mem
    buffers_->SetHostInputBuffer(images);
    buffers_->CopyInputToDevice();
    // Synchronously execute inference a network.
    bool status = context_->executeV2(buffers_->GetDeviceBindings().data());
    if (!status)
    {
        return nullptr;
    }

    // Memcpy from device output buffers to host output buffers
    buffers_->CopyOutputToHost();
    // return output host mem
    return buffers_->GetHostOutputBuffer();
}

void *Infer::Inference(const cv::Mat &images)
{
    // Memcpy from host input buffers to device input buffers
    // set mem to input host
    // copy host mem to device mem

    buffers_->SetHostInputBuffer(images);

    buffers_->CopyInputToDevice();

    // Synchronously execute inference a network.
    bool status = context_->executeV2(buffers_->GetDeviceBindings().data());
    if (!status)
    {
        return nullptr;
    }

    // Memcpy from device output buffers to host output buffers
    buffers_->CopyOutputToHost();

    // return output host mem
    return buffers_->GetHostOutputBuffer();
}

//异步推理
void *Infer::InferenceAsync(std::vector<cv::Mat> images)
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
    return buffers_->GetHostOutputBuffer();
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

    // //获取模型属性 batch size min pro max
    GetModelProperty();
    buffers_.reset(new EngineBufferManager(engine_, min_batch_size_));
    input_buffer_size_ = buffers_->Size(input_tensort_name_);
    output_buffer_size_ = buffers_->Size(output_tensort_name_);
    model_status_ = true;
    return true;
}

//获取模型属性  input_name output_name batch
void Infer::GetModelProperty()
{
    for (int i = 0; i < engine_->getNbBindings(); i++)
    {
        auto dims = engine_->getBindingDimensions(i);

        if (engine_->bindingIsInput(i))
        {
            // 动态batch
            if (dims.d[0] <= 0)
            {
                int32_t profiles_number = engine_->getNbOptimizationProfiles();
                auto dims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);
                min_batch_size_ = dims.d[0];
                dims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT);
                opt_batch_size_ = dims.d[0];
                dims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                max_batch_size_ = dims.d[0];
            }
            else
            {
                min_batch_size_ = dims.d[0];
                opt_batch_size_ = dims.d[0];
                max_batch_size_ = dims.d[0];
            }
            input_dims_ = dims;
            input_tensort_name_ = engine_->getBindingName(i);
            XF_LOGT(INFO, TAG, "input tensort dim  %s", dims_str(dims).c_str());
        }
        else
        {
            output_dims_ = dims;
            output_tensort_name_ = engine_->getBindingName(i);
            XF_LOGT(INFO, TAG, "output tensort dim  %s", dims_str(dims).c_str());
        }
    }
    XF_LOGT(INFO, TAG, "min batch size [%d]", min_batch_size_);
    XF_LOGT(INFO, TAG, "opt batch size [%d]", opt_batch_size_);
    XF_LOGT(INFO, TAG, "max batch size [%d]", max_batch_size_);
    XF_LOGT(INFO, TAG, "input tensort name [%s]", input_tensort_name_.c_str());
    XF_LOGT(INFO, TAG, "output tensort name [%s]", output_tensort_name_.c_str());
}
std::vector<int> Infer::GetOutputShape()
{
    std::vector<int> shape;

    for (int i = 1; i < output_dims_.nbDims; i++)
    {
        shape.push_back(output_dims_.d[i]);
    }
    return shape;
}
std::vector<int> Infer::GetInputShape()
{
    std::vector<int> shape;

    for (int i = 1; i < input_dims_.nbDims; i++)
    {
        shape.push_back(input_dims_.d[i]);
    }
    return shape;
}

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
void Infer::TestModel2()
{
    std::string engine_file("./models/yolov5n.onnx.engine");
    LoadModel(engine_file);

    std::string image_file = "images/input-image.jpg";
    auto image = cv::imread(image_file);
    std::vector<cv::Mat> images;
    images.push_back(image);
    auto output = Inference(images);
}
void Infer::TestModel()
{
    std::string onnx_file("./models/yolov5n.onnx");
    std::string engine_file("./models/yolov5n.onnx.engine");
    trt::builder_engine(trt::FP16, 1, onnx_file, engine_file);

    TRTLogger logger;

    auto engine_data = LoadEngine(engine_file);
    auto runtime = make_shared_ptr(nvinfer1::createInferRuntime(logger));
    auto engine = make_shared_ptr(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    if (engine->getNbBindings() != 2)
    {
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbBindings() - 1);
        return;
    }
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_shared_ptr(engine->createExecutionContext());

    printf("engine->getBindingName(0) %s\n", engine->getBindingName(0));
    printf("engine->getName %s\n", engine->getName());
    auto dims = engine->getBindingDimensions(0);
    int input_batch = dims.d[0];
    int input_channel = dims.d[1];
    int input_height = dims.d[2];
    int input_width = dims.d[3];
    printf("input_batch %d input_channel %d input_height %d input_width %d\n", input_batch, input_channel, input_height,
           input_width);

    int input_numel = input_batch * input_channel * input_height * input_width;
    float *input_data_host = nullptr;
    float *input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    std::string image_file = "images/input-image.jpg";
    auto image = cv::imread(image_file);
    int image_area = image.cols * image.rows;
    unsigned char *pimage = image.data;
    float *phost_b = input_data_host + image_area * 0;
    float *phost_g = input_data_host + image_area * 1;
    float *phost_r = input_data_host + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
    // memcpy(input_data_host, image.data, image_area * 3);
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    auto output_dims = engine->getBindingDimensions(1);
    int output_batch_size = output_dims.d[0];
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];

    printf("output_batch_size %d, output_numbox %d, output_numprob %d \n", output_batch_size, output_numbox,
           output_numprob);
    int num_classes = output_numprob - 5;
    int output_numel = output_batch_size * output_numbox * output_numprob;
    float *output_data_host = nullptr;
    float *output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float *bindings[] = {input_data_device, output_data_device};
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel,
                                 cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // decode box：从不同尺度下的预测狂还原到原输入图上(包括:预测框，类被概率，置信度）
    std::vector<std::vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;
    for (int i = 0; i < output_numbox; ++i)
    {
        float *ptr = output_data_host + i * output_numprob;
        float objness = ptr[4];
        if (objness < confidence_threshold)
            continue;

        float *pclass = ptr + 5;
        int label = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob = pclass[label];
        float confidence = prob * objness;
        if (confidence < confidence_threshold)
            continue;

        // 中心点、宽、高
        float cx = ptr[0];
        float cy = ptr[1];
        float width = ptr[2];
        float height = ptr[3];

        // 预测框
        float left = cx - width * 0.5;
        float top = cy - height * 0.5;
        float right = cx + width * 0.5;
        float bottom = cy + height * 0.5;

        bboxes.push_back({left, top, right, bottom, (float)label, confidence});
    }
    printf("decoded bboxes.size = %d\n", bboxes.size());

    // nms非极大抑制
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float> &a, std::vector<float> &b) { return a[5] > b[5]; });
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float> &a, const std::vector<float> &b) {
        float cross_left = std::max(a[0], b[0]);
        float cross_top = std::max(a[1], b[1]);
        float cross_right = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) +
                           std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if (cross_area == 0 || union_area == 0)
            return 0.0f;
        return cross_area / union_area;
    };

    for (int i = 0; i < bboxes.size(); ++i)
    {
        if (remove_flags[i])
            continue;

        auto &ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for (int j = i + 1; j < bboxes.size(); ++j)
        {
            if (remove_flags[j])
                continue;

            auto &jbox = bboxes[j];
            if (ibox[4] == jbox[4])
            {
                // class matched
                if (iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    printf("box_result.size = %d\n", box_result.size());

    for (int i = 0; i < box_result.size(); ++i)
    {
        auto &ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        cv::Scalar color;

        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0), 3);
    }
    std::string output_image_file = "output-image.jpg";
    cv::imwrite(output_image_file, image);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}
} // namespace trt
