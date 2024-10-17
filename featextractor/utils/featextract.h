/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 09:34:08
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-03 13:32:44
 * @Description: 特征提取代码
 */
#ifndef FEATEXTRACT_H
#define FEATEXTRACT_H

#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "common.h"
#include <fstream>
#include <iostream>

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

class FeatExtractModel {
public:
    FeatExtractModel();
    ~FeatExtractModel();
    bool loadModel(const std::string engine_name);
    bool inference(cv::Mat frame, std::vector<float>& feat);
    bool batchInference(std::vector<cv::Mat> img_frames, std::vector<std::vector<float>>& feats);

private:
    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;
    int m_featdim;

    bool deserializeEngine(const std::string engine_name);
    bool doInference(const std::vector<cv::Mat> img_batch, std::vector<std::vector<float>>& feat);
    void preProcess(std::vector<cv::Mat> frame, std::vector<float>& data);

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