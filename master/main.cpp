#include "common/logging.h"
#include "config_env.h"
#include "inference/inference.h"
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
    auto infer = xf::create_inference("a");
    if (infer != nullptr)
        infer->Forward();
}