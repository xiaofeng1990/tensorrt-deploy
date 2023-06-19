#include "yolo_infer.h"
#include "common/common.h"
#include "common/config_helper.h"
#include "common/logging.h"
#include "infer_interface.h"
#include "tensorrt/builder.h"
#include "tensorrt/infer.h"
#include <future>
#include <mutex>
#include <queue>
#include <thread>
namespace xf {

class YoloImpl : public InferInterface {

  public:
    virtual ~YoloImpl();
    bool LoadModel(const std::string &onnx_file);
    virtual std::shared_future<std::vector<Box>> Commits(const std::string &image_file) override;
    void Destroy();
    void Preprocess(Job &job);
    void Postprocess();
    //推理消费
    void Worker(std::string onnx_file, std::promise<bool> &pro);
    std::vector<Box> Decode(float *predict, int num_box, int num_prob, const float *d2i,
                            float confidence_threshold = 0.25f, float nms_threshold = 0.45f);

  private:
    //消费者
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::queue<Job> qjobs_;
    std::mutex lock_job_;
    std::condition_variable condition_;
    int max_batch_size_;
    trt::Infer infer_;
    // channel, height, width 3x640x640
    std::vector<int> input_shape_;
    // numbox, numprob 25200x85
    std::vector<int> output_shape_;
    static constexpr const char *TAG = "yolo_infer";
};
YoloImpl::~YoloImpl()
{
    running_ = false;
    condition_.notify_one();
    if (thread_.joinable())
    {
        thread_.join();
    }
}

bool YoloImpl::LoadModel(const std::string &engine_file)
{
    //资源哪里分配，哪里释放，哪里使用
    std::promise<bool> pro;
    running_ = true;
    // todo 优化为线程池
    thread_ = std::thread(&YoloImpl::Worker, this, engine_file, std::ref(pro));

    return pro.get_future().get();
}

void YoloImpl::Worker(std::string engine_file, std::promise<bool> &pro)
{
    //模型加载
    bool ret = infer_.LoadModel(engine_file);

    if (!ret)
    {
        XF_LOGT(ERROR, TAG, "load model %s failed\n", engine_file.c_str());
        pro.set_value(false);
        return;
    }
    else
    {
        XF_LOGT(INFO, TAG, "load model %s succeed\n", engine_file.c_str());
        max_batch_size_ = infer_.GetMaxBatchSize();
        input_shape_ = infer_.GetInputShape();
        output_shape_ = infer_.GetOutputShape();

        XF_LOGT(INFO, TAG, "max batch size %d\n", max_batch_size_);
        pro.set_value(true);
    }

    //模型使用
    std::vector<Job> jobs;
    int batch_id = 0;
    int output_numbox = output_shape_[0];
    int output_numprob = output_shape_[1];
    while (running_)
    {
        std::unique_lock<std::mutex> lock(lock_job_);
        condition_.wait(lock, [&]() {
            // true return
            // false wait
            return !qjobs_.empty() || !running_;
        });
        if (!running_)
        {
            XF_LOGT(INFO, TAG, "stop inference thread\n");
            break;
        }

        // pop data into queue
        while (jobs.size() < max_batch_size_ && !qjobs_.empty())
        {
            jobs.push_back(qjobs_.front());
            qjobs_.pop();
        }
        // set input host data
        std::vector<cv::Mat> images;
        for (int i = 0; i < jobs.size(); i++)
        {
            // load image
            images.push_back(jobs[i].warp_mat);
            // todo 前处理
        }

        auto outputs = static_cast<float *>(infer_.Inference(images));

        // 后处理

        for (int i = 0; i < jobs.size(); i++)
        {
            auto &job = jobs[i];
            // * output size: 5x25200x85
            auto predict = (float *)(outputs + output_numbox * output_numprob * i);
            //后处理
            auto boxes = Decode(predict, output_numbox, output_numprob, job.d2i);
            job.pro->set_value(boxes);
        }
        batch_id++;
        jobs.clear();
    }
    XF_LOGT(INFO, TAG, "thread down\n");
    // delete model
    XF_LOGT(INFO, TAG, "release model %s\n", engine_file.c_str());
    // context.clear();
}

std::shared_future<std::vector<Box>> YoloImpl::Commits(const std::string &image_file)
{
    // push data into queue
    Job job;
    job.pro.reset(new std::promise<std::vector<Box>>());

    // todo 预处理
    job.image_file = image_file;

    Preprocess(job);

    std::lock_guard<std::mutex> lock(lock_job_);
    qjobs_.push(job);
    // detecton push
    // face push
    // feature push
    // detection face feature 并发执行
    condition_.notify_one();
    return job.pro->get_future();

    // XF_LOGT(INFO, TAG, "using %s inference\n", context_.c_str());
}

void YoloImpl::Preprocess(Job &job)
{
    // warpAffine
    int input_channel = input_shape_[0];
    int input_height = input_shape_[1];
    int input_width = input_shape_[2];
    cv::Mat image = cv::imread(job.image_file);

    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6];
    /*
        M = [
                scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
                0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
                0,        0,                     1
            ]
    */
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-scale * image.cols + input_width + scale - 1) * 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);     // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, job.d2i); // dst to image, 2x3 matrix
    // 获得逆矩阵
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat warpt_image(input_height, input_width, CV_8UC3);
    // 对图像做平移缩放旋转变换,可逆
    cv::warpAffine(image, warpt_image, m2x3_i2d, warpt_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar::all(114));
    // std::string warp_image_file = "./images/input-image.jpg";
    // cv::imwrite(warp_image_file, input_image);
    job.warp_mat = warpt_image;
}
void YoloImpl::Postprocess() {}

void YoloImpl::Destroy() {}

std::vector<Box> YoloImpl::Decode(float *predict, int num_box, int num_prob, const float *d2i,
                                  float confidence_threshold, float nms_threshold)
{
    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::microseconds>(systemtime.time_since_epoch()).count());

    // cx, cy, width, height, objness, classification*80
    // 一行是85列
    std::vector<Box> boxes;
    int num_classes = num_prob - 5;
    // 第一个循环，根据置信度挑选box
    for (int i = 0; i < num_box; i++)
    {
        // 获得每一行的首地址
        float *pitem = predict + i * num_prob;
        // 获取当前网格有目标的置信度
        float objness = pitem[4];
        if (objness < confidence_threshold)
            continue;

        // 获取类别置信度的首地址
        float *pclass = pitem + 5;
        // std::max_element 返回从pclass到pclass+num_classes中最大值的地址，
        // 减去 pclass 后就是索引
        int label = std::max_element(pclass, pclass + num_classes) - pclass;
        // 分类目标概率
        float prob = pclass[label];
        // 当前网格中有目标，且为某一个类别的的置信度
        float confidence = prob * objness;

        if (confidence < confidence_threshold)
            continue;

        float cx = pitem[0];
        float cy = pitem[1];
        float width = pitem[2];
        float height = pitem[3];

        float left = cx - width * 0.5;
        float top = cy - height * 0.5;
        float right = cx + width * 0.5;
        float bottom = cy + height * 0.5;

        // 对应图上的位置
        left = d2i[0] * left + d2i[2];
        right = d2i[0] * right + d2i[2];
        top = d2i[0] * top + d2i[5];
        bottom = d2i[0] * bottom + d2i[5];
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }
    // 对所有box根据置信度排序
    std::sort(boxes.begin(), boxes.end(), [](Box &a, Box &b) { return a.confidence > b.confidence; });
    // 记录box是否被删除，被删除为true
    std::vector<bool> remove_flags(boxes.size());
    // 保存box
    std::vector<Box> box_result;
    box_result.reserve(boxes.size());

    for (int i = 0; i < boxes.size(); i++)
    {
        if (remove_flags[i])
            continue;
        auto &ibox = boxes[i];
        box_result.emplace_back(ibox);
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (remove_flags[j])
                continue;
            auto &jbox = boxes[j];
            if (ibox.label == jbox.label)
            {
                if (iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::microseconds>(systemtime.time_since_epoch()).count());

    printf("cpu yolov5 postprocess %ld ns\n", timestamp2 - timestamp1);

    return box_result;
}

// create instance and load model
std::shared_ptr<InferInterface> create_inference(trt::InferConfig infer_config)
{
    //通过onnx file build engine文件
    //生成engine文件名称
    auto folder = xffw::GetDirectory(infer_config.model_path);
    auto filename = xffw::GetFileNameFromPath(infer_config.model_path);
    std::string engine_name = folder + "/" + filename + ".engine";

    //生成engine文件
    trt::builder_engine(trt::FP16, infer_config.max_batch_size, infer_config.model_path, engine_name);

    std::shared_ptr<YoloImpl> instance(new YoloImpl());
    if (!instance->LoadModel(engine_name))
        instance.reset();
    return instance;
}

float iou(const Box &a, const Box &b)
{
    float cross_left = std::max(a.left, b.left);
    float cross_top = std::max(a.top, b.top);
    float cross_right = std::min(a.right, b.right);
    float cross_bottom = std::min(a.bottom, b.bottom);

    float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
    float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) +
                       std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
    if (cross_top == 0 || union_area == 0)
        return 0.0f;

    return cross_area / union_area;
}

} // namespace xf
