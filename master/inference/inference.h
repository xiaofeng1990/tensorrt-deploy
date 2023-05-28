#ifndef __MASTER_INTERENCE_INFERENCE_H_
#define __MASTER_INTERENCE_INFERENCE_H_
#include "infer_interface.h"
#include <memory>
#include <string>
namespace xf {

std::shared_ptr<InferInterface> create_inference(const std::string &file);

} // namespace xf

#endif // __MASTER_INTERENCE_INFERENCE_H_