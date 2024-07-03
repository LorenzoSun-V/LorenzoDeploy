/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 09:56:53
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-03 09:50:09
 * @Description:  YOLOv10模型前处理、推理、后处理代码
 */
#ifndef YOLOV10MODELMANAGER_H
#define YOLOV10MODELMANAGER_H

#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "common.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
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

class YOLOV10ModelManager {
public:
    YOLOV10ModelManager();
    ~YOLOV10ModelManager();
    bool loadModel(const std::string engine_name);
    bool inference(cv::Mat frame, std::vector<DetBox>& detBoxs);
    bool batchInference(std::vector<cv::Mat> img_frames, std::vector<std::vector<DetBox>>& batchDetBoxes);

private:
    const int kOutputSize = 300;  // defulat output of YOLOv10 is 300
    std::vector<float> factor;      // the factor(calculate in the preProcess) is used to adjust bbox size in the postProcess
    float conf = 0.25;
    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;

    bool deserializeEngine(const std::string engine_name);
    bool doInference(const std::vector<cv::Mat> img_batch, std::vector<std::vector<DetBox>>& batchDetBoxes);
    void preProcess(std::vector<cv::Mat> frame, std::vector<float>& data);
    bool postProcess(float* result, std::vector<std::vector<DetBox>>& detBoxs);

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

#endif // YOLOV10MODEL_H
