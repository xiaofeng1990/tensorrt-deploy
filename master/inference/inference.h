#ifndef __MASTER_INTERENCE_INFERENCE_H_
#define __MASTER_INTERENCE_INFERENCE_H_
#include "infer_interface.h"
#include <future>
#include <memory>
#include <string>
namespace xf {
#define DEFAULT_MAX_BATCH_SIZE 5
struct Job {
    std::shared_ptr<std::promise<std::string>> pro;
    std::string input;
};

std::shared_ptr<InferInterface> create_inference(const std::string &file);

} // namespace xf

#endif // __MASTER_INTERENCE_INFERENCE_H_