/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 09:56:53
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-13 10:45:38
 * @Description:  YOLOv8OBB模型前处理、推理、后处理代码
 */
#ifndef YOLOV8OBBMODEL_H
#define YOLOV8OBBMODEL_H

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

class YOLOV8OBBModel {
public:
    YOLOV8OBBModel();
    ~YOLOV8OBBModel();
    bool loadModel(const std::string engine_name);
    bool inference(cv::Mat frame, std::vector<DetBox>& result);
    bool batch_inference(std::vector<cv::Mat> batch_images, std::vector<std::vector<DetBox>>& batch_result);
private:
    int m_kOutputSize;  
    int m_kInputSize;
    int m_kDecodeSize; 

    model_param_t m_model;
    trt_param_t m_trt;

    bool deserializeEngine(const std::string engine_name);

    bool doInference(std::vector<cv::Mat> img_batch);
    
    int kMaxInputImageSize;

    float* inputSrcDevice;
    float* outputSrcDevice;

    float* decode_ptr_host = nullptr;    // CPU用于接收GPU上置信度筛选和NMS后的结果
    float* decode_ptr_device = nullptr;  // 用于存放解码后的结果，进行置信度筛选和NMS
    
    std::vector<float> inputData;
    std::vector<float> output_data;
};

#endif // YOLOV10MODEL_H