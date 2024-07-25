#ifndef YOLOV5MODEL_H
#define YOLOV5MODEL_H

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <utility>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"
#include "common.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YOLOV5Model 
{
public:
    YOLOV5Model();
    ~YOLOV5Model();
    bool loadModel(const std::string& engine_name);
    bool inference(cv::Mat& frame, std::vector<DetBox>& detBoxs);
    bool batchInference(std::vector<cv::Mat>& batchframes, std::vector<std::vector<DetBox>>& batchDetBoxs);

private:
    bool deserializeEngine(const std::string& engine_name);
    bool prepareBuffer();
    void doInference(std::vector<cv::Mat> img_batch, std::vector<std::vector<Detection>>& res_batch); 

    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;
    // Deserialize the engine from file
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    
    float* device_buffers[2];
    float* cpu_output_buffer; 
    cudaStream_t stream;
    Logger gLogger;
    int kOutputSize;
};

#endif // YOLOV8MODEL_H
