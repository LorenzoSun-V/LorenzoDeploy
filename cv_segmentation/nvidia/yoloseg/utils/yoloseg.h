/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-26 08:51:19
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-06 09:42:19
 * @Description: 
 */
#ifndef YOLOSegModel_H
#define YOLOSegModel_H

#include <fstream>
#include <sys/stat.h>
#include <common.h>

#include "cuda.h"
#include "NvInfer.h"
#include "preprocess.h"
#include "postprocess.h"

using namespace nvinfer1;

//nvinfer1::ILogger
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

typedef struct {
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
}trt_param_t;

class YOLOSegModel 
{
public:
    YOLOSegModel();
    ~YOLOSegModel();
    bool loadModel(const std::string engine_name, bool bUseYOLOv8);
    bool inference(cv::Mat frame, std::vector<SegBox>& result, std::vector<cv::Mat>& masks);
    bool batch_inference(std::vector<cv::Mat> batch_images, std::vector<std::vector<SegBox>>& batch_result, std::vector<std::vector<cv::Mat>>& batch_masks);

private:
    bool deserializeEngine(const std::string engine_name);
    bool doInference(std::vector<cv::Mat> img_batch);

    trt_param_t m_trt;
    model_param_t m_model;
    
    bool yolov8;  // 是否是 YOLOv8 模型，true为 YOLOv8，false为 YOLOv5，两种模型的输出内容不同
    
    float* gpu_input_data;
    float* gpu_det_output_data;
    float* gpu_seg_output_data;
    
    std::vector<float> cpu_input_data;
    std::vector<float> cpu_det_output_data;
    std::vector<float> cpu_seg_output_data;

    int m_kDetOutputSize;
    int m_kSegOutputSize;
    int kMaxInputImageSize;
};

#endif // YOLOSegModel_H
