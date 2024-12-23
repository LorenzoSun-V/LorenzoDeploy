/*
 * @Author: BTZN0323 jiajunjie@boton-tech.com
 * @Date: 2024-12-12 09:34:08
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-20 15:21:01
 * @Description: 图像分类代码
 */
#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <fstream>
#include "postprocess.h"


using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class ClasssificationModel {
public:
    ClasssificationModel();
    ~ClasssificationModel();
    bool loadModel(const std::string engine_name);
    bool inference(cv::Mat frame, std::vector<ClsResult>& cls_rets, int topK);

private:
    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;
    int m_classnum;

    bool deserializeEngine(const std::string engine_name);
    void preProcess(std::vector<cv::Mat> frame, std::vector<float>& image_data);
    bool doInference(const std::vector<cv::Mat> img_batch, std::vector<std::vector<float>>& result_data);

    // Deserialize the engine from file
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    void* inputSrcDevice;
    void* outputSrcDevice;
    std::vector<float> inputData;
    std::vector<float> output_data;
};


#endif