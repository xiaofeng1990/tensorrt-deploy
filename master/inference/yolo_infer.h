#ifndef __MASTER_INTERENCE_YOLO_INFER_H_
#define __MASTER_INTERENCE_YOLO_INFER_H_

#include "infer_interface.h"
#include "tensorrt/trt_utils.h"
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
namespace xf {
#define DEFAULT_MAX_BATCH_SIZE 5

typedef std::vector<Box> BoxArray;

struct Job {
    std::shared_ptr<std::promise<std::vector<Box>>> pro;
    std::string image_file;
    cv::Mat warp_mat;
    // dst to image 2*3 matrix
    float d2i[6];
};

std::shared_ptr<InferInterface> create_inference(trt::InferConfig infer_config);
float iou(const Box &a, const Box &b);
} // namespace xf

#endif // __MASTER_INTERENCE_YOLO_INFER_H_