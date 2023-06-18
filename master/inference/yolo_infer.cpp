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
    void Preprocess(std::vector<Job> &jobs);
    void Postprocess();
    //推理消费
    void Worker(std::string onnx_file, std::promise<bool> &pro);
    std::vector<Box> Decode(float *predict, int num_box, int num_prob, float confidence_threshold = 0.25f,
                            float nms_threshold = 0.45f);

  private:
    //消费者
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::queue<Job> qjobs_;
    std::mutex lock_job_;
    std::condition_variable condition_;
    int max_batch_size_;
    trt::Infer infer_;
    std::vector<int> input_shape_;
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
            images.push_back(jobs[i].image);
            // todo 前处理
        }

        auto outputs = static_cast<float *>(infer_.Inference(images));

        // 后处理

        for (int i = 0; i < jobs.size(); i++)
        {
            auto &job = jobs[i];
            // * output size: 5x25200x85
            auto predict = (float *)(outputs + output_numbox * output_numprob * i);

            auto boxes = Decode(predict, output_numbox, output_numprob);

            for (auto &box : boxes)
            {
                cv::rectangle(job.image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom),
                              cv::Scalar(0, 255, 0), 2);
                cv::putText(job.image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8,
                            cv::Scalar(0, 0, 255), 2, 16);
            }
            std::string save_image_file = "image-draw.jpg";
            cv::imwrite(save_image_file, job.image);
            //后处理
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
    job.image = cv::imread(image_file);

    job.image_file = image_file;
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

void YoloImpl::Preprocess(std::vector<Job> &jobs)
{
    // warpAffine
    //计算矩阵
    //矩阵变换
}
void YoloImpl::Postprocess() {}

void YoloImpl::Destroy() {}

std::vector<Box> YoloImpl::Decode(float *predict, int num_box, int num_prob, float confidence_threshold,
                                  float nms_threshold)
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
