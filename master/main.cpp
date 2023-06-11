#include "common/config_helper.h"
#include "common/logging.h"
#include "config_env.h"
#include "inference/inference.h"

#include "tensorrt/buffers.h"
#include "tensorrt/builder.h"
#include "tensorrt/cuda_tools.h"
#include "version.h"
#include <fstream>
#include <iostream>
#include <string>
std::vector<unsigned char> load_file(const std::string file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
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

int main()
{
    trt::Logger gLogger;
    XF_LOG(INFO, "ai serving version: %s", XF_VERSION);
    xf::ConfigEnv config_env;
    config_env.Init();
    XF_LOG(INFO, "Hello TensorRT");
    XF_LOG(INFO, "This is a TensorRT deploy project");

    // config log
    xffw::XfConfig *config = xffw::ConfigHelper::Ins();

    std::string model_file;
    config->Get("inference", "model_file", model_file);
    std::string engine_file("./models/yolov5s.engine");

    auto engine_data = load_file(engine_file);
    auto runtime = trt::make_shared_ptr(nvinfer1::createInferRuntime(gLogger));

    auto engine = trt::make_shared_ptr(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return -1;
    }
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    trt::BufferManager buffers(engine);
    auto context = trt::make_shared_ptr(engine->createExecutionContext());

    return 0;
}

//  auto infer = xf::create_inference("a");
//     if (infer == nullptr)
//     {
//         XF_LOG(INFO, "load model failed!");
//         return -1;
//     }

//     auto infera = xf::create_inference("a");
//     if (infera == nullptr)
//     {
//         XF_LOG(INFO, "load model failed!");
//         return -1;
//     }

//     //并行
//     auto fa = infer->Forward("a");
//     auto fb = infer->Forward("b");
//     auto fc = infer->Forward("c");
//     auto fd = infer->Forward("d");
//     printf("%s \n", fa.get().c_str());
//     printf("%s \n", fb.get().c_str());
//     printf("%s \n", fc.get().c_str());
//     printf("%s \n", fd.get().c_str());

//     //串行
//     auto faa = infera->Forward("a").get();
//     auto fab = infera->Forward("b").get();
//     auto fac = infera->Forward("c").get();
//     auto fad = infera->Forward("d").get();
//     printf("%s \n", faa.c_str());
//     printf("%s \n", fab.c_str());
//     printf("%s \n", fac.c_str());
//     printf("%s \n", fad.c_str());