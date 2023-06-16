#include "common/config_helper.h"
#include "common/logging.h"
#include "config_env.h"
#include "inference/yolo_infer.h"
#include "tensorrt/builder.h"
#include "tensorrt/cuda_tools.h"
#include "tensorrt/engine_buffers.h"
#include "tensorrt/trt_logger.h"
#include "tensorrt/trt_utils.h"
// #include "version.h"
#include <fstream>
#include <iostream>
#include <string>

int main()
{
    // config log
    xffw::XfConfig *config = xffw::ConfigHelper::Ins();

    trt::InferConfig infer_config;

    config->Get("inference", "model_file", infer_config.model_path);

    config->Get("inference", "max_batch_size", infer_config.max_batch_size);
    config->Get("inference", "confidence_threshold", infer_config.conf_threshold);
    config->Get("inference", "iou_threshold", infer_config.iou_threshold);
    auto infer = xf::create_inference(infer_config);

    return 0;
}

// trt::Logger gLogger;
//     XF_LOG(INFO, "ai serving version: %s", XF_VERSION);
//     xf::ConfigEnv config_env;
//     config_env.Init();
//     XF_LOG(INFO, "Hello TensorRT");
//     XF_LOG(INFO, "This is a TensorRT deploy project");

//     // config log
//     xffw::XfConfig *config = xffw::ConfigHelper::Ins();

//     std::string model_file;
//     config->Get("inference", "model_file", model_file);
//     std::string engine_file("./models/yolov5s.engine");

//     auto engine_data = load_file(engine_file);
//     auto runtime = trt::make_shared_ptr(nvinfer1::createInferRuntime(gLogger));

//     auto engine = trt::make_shared_ptr(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
//     if (engine == nullptr)
//     {
//         printf("Deserialize cuda engine failed.\n");
//         runtime->destroy();
//         return -1;
//     }
//     cudaStream_t stream = nullptr;
//     checkRuntime(cudaStreamCreate(&stream));
//     trt::BufferManager buffers(engine);
//     auto context = trt::make_shared_ptr(engine->createExecutionContext());

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