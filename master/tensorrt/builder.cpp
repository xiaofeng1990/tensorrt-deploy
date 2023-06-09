#include "builder.h"
#include "common/common.h"
#include "common/logging.h"
#include "common/timer.h"
#include "trt_logger.h"
#include "trt_utils.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
namespace trt {

static TRTLogger gLogger;
static constexpr const char *TAG = "builder";

bool builder_engine(Mode mode, unsigned int max_batch_size, const std::string onnx_file, const std::string engine_file,
                    const size_t max_workspace_size)
{
    //计算onnx文件是否变化，如果变化，重新生成engine文件
    if (!compare_md5_is_diff(onnx_file) && exists(engine_file.c_str()))
    {
        XF_LOGT(INFO, TAG, "engine file %s has exists.\n", engine_file.c_str());
        return true;
    }
    XF_LOGT(INFO, TAG, "engine file name %s\n", engine_file.c_str());
    // creat builder
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));

    // creat network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    // creat config
    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(max_workspace_size);
    if (mode == Mode::FP16)
    {
        if (!builder->platformHasFastFp16())
        {
            XF_LOGT(WARN, TAG, "Platform not support fast fp16");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // load onnxfile
    auto onnx_parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!onnx_parser->parseFromFile(onnx_file.c_str(), 1))
    {
        XF_LOGT(ERROR, TAG, "Can not parse onnx file %s", onnx_file.c_str());
        return false;
    }
    // profile
    auto profile = builder->createOptimizationProfile();
    int input_number = network->getNbInputs();
    int output_number = network->getNbOutputs();
    // set min, max, opt batch
    for (int i = 0; i < input_number; i++)
    {
        auto input = network->getInput(i);
        auto input_dims = input->getDimensions();
        XF_LOGT(INFO, TAG, "input tensor dims  %s", dims_str(input_dims).c_str());
        int batch = input_dims.d[0];
        if (batch <= 0)
        {
            input_dims.d[0] = 1;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
            input_dims.d[0] = max_batch_size;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
        }
    }
    config->addOptimizationProfile(profile);

    // save engine file
    XF_LOGT(INFO, TAG, "Building engine...");
    xffw::Timer timer;
    timer.Start();
    auto engine = UniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr)
    {
        XF_LOGT(ERROR, TAG, "engine is nullptr");
        return false;
    }
    XF_LOGT(INFO, TAG, "Build engine done %lld s !", timer.Stop() / 1000);
    // serialize the engine, then close everything down
    auto seridata = UniquePtr<nvinfer1::IHostMemory>(engine->serialize());
    // save file
    // 计算engine 文件的 md5
    return save_engine_file(engine_file, seridata->data(), seridata->size());
}

const char *mode_to_string(Mode type)
{
    switch (type)
    {
    case Mode::FP16:
        return "FP16";
    case Mode::FP32:
        return "FP32";
    case Mode::INT8:
        return "INT8";
    default:
        return "UnknowTRTMode";
    }
}

bool save_engine_file(const std::string engine_file, const void *data, size_t length)
{
    FILE *f = fopen(engine_file.c_str(), "wb");
    if (!f)
    {
        XF_LOGT(ERROR, TAG, "save [open] engine file %s failed", engine_file.c_str());
        return false;
    }
    if (data && length > 0)
    {
        auto ret = fwrite(data, 1, length, f);
        if (ret != length)
        {
            XF_LOGT(ERROR, TAG, "save [write] engine file %s failed", engine_file.c_str());
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

bool compare_md5_is_diff(std::string model_path)
{
    const auto md5 = xffw::file_md5(model_path);
    XF_LOGT(DEBUG, TAG, "file: %s, md5: %s", model_path.c_str(), md5.c_str());
    if (md5.empty())
    {
        return true;
    }

    const std::string model_md5_file = model_path + ".md5";
    std::string exists_md5;
    std::ifstream is(model_md5_file);
    if (is.is_open())
    {
        is >> exists_md5;
        XF_LOGT(DEBUG, TAG, "exists md5: %s", exists_md5.c_str());
        is.close();
        // Do not re-build engine for same model file
        if (md5.compare(exists_md5) == 0)
        {
            return false;
        }
    }
    // save md5
    std::ofstream os;
    os.open(model_md5_file, std::ios::out);
    if (!os.is_open())
    {
        XF_LOGT(ERROR, TAG, "Open file(%s) fail, To re-build engine(1)", model_md5_file.c_str());
        return true;
    }
    os << md5;
    os.close();

    return true;
}

} // namespace trt
