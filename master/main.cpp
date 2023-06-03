#include "common/config_helper.h"
#include "common/logging.h"
#include "config_env.h"
#include "inference/inference.h"
#include "tensorrt/builder.h"
#include "version.h"
#include <iostream>
#include <string>

int main()
{
    XF_LOG(INFO, "ai serving version: %s", XF_VERSION);
    xf::ConfigEnv config_env;
    config_env.Init();
    XF_LOG(INFO, "Hello TensorRT");
    XF_LOG(INFO, "This is a TensorRT deploy project");

    // config log
    xffw::XfConfig *config = xffw::ConfigHelper::Ins();

    std::string model_file;
    config->Get("inference", "model_file", model_file);
    trt::Builder builder;
    builder.Compile(trt::Mode::FP16, 5, model_file, "./yolov5s.engine");
    auto infer = xf::create_inference("a");
    if (infer == nullptr)
    {
        XF_LOG(INFO, "load model failed!");
        return -1;
    }

    auto infera = xf::create_inference("a");
    if (infera == nullptr)
    {
        XF_LOG(INFO, "load model failed!");
        return -1;
    }

    //并行
    auto fa = infer->Forward("a");
    auto fb = infer->Forward("b");
    auto fc = infer->Forward("c");
    auto fd = infer->Forward("d");
    printf("%s \n", fa.get().c_str());
    printf("%s \n", fb.get().c_str());
    printf("%s \n", fc.get().c_str());
    printf("%s \n", fd.get().c_str());

    //串行
    auto faa = infera->Forward("a").get();
    auto fab = infera->Forward("b").get();
    auto fac = infera->Forward("c").get();
    auto fad = infera->Forward("d").get();
    printf("%s \n", faa.c_str());
    printf("%s \n", fab.c_str());
    printf("%s \n", fac.c_str());
    printf("%s \n", fad.c_str());
    return 0;
}