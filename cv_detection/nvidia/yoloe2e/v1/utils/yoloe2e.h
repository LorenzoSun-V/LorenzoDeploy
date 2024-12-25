#ifndef YOLOE2EMODELMANAGER_H
#define YOLOE2EMODELMANAGER_H

#include "cuda.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

#include "common.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <array>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
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

class YOLOE2EModelManager {
public:
    YOLOE2EModelManager();
    ~YOLOE2EModelManager();
    bool loadModel(const std::string engine_name);
    bool inference(cv::Mat frame, std::vector<DetBox>& detBoxs);

private:
    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;

    bool deserializeEngine(const std::string engine_name);
    bool doInference(cv::Mat img, std::vector<DetBox>& batchDetBoxes);
    float letterbox( const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, 
    int stride, const cv::Scalar& color,  bool fixed_shape, bool scale_up);
    float* blobFromImage(cv::Mat& img);

   // void preProcess(cv::Mat img, std::vector<float>& data);
    void postProcess(cv::Mat img, float scale, std::vector<DetBox>& detBoxs);

    // Deserialize the engine from file
    Logger gLogger;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;

    void* buffs[5];
    int  in_size, num_size1, boxes_size2, scores_size3, classes_size4;
    int* num_dets;
    float* det_boxes;
    float* det_scores;
    int* det_classes;
    float* blob;
};

#endif // YOLOE2EMODEL_H
