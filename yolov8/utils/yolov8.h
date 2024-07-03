#ifndef YOLOV8MODELMANAGER_H
#define YOLOV8MODELMANAGER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>

#include "common.h"
#include "model.h"
#include "preprocess.h"
#include "postprocess.h"
#include "cuda_utils.h"
#include "logging.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;

class YOLOV8ModelManager {
public:
    YOLOV8ModelManager();
    ~YOLOV8ModelManager();
    bool loadModel(const std::string& engine_name);
    bool inference(cv::Mat& frame, std::vector<DetBox>& detBoxs);
    bool batchInference(std::vector<cv::Mat>& imglist, std::vector<std::vector<DetBox>>& batchDetBoxs);

private:
    bool deserializeEngine(const std::string& engine_name);
    bool prepareBuffer();
    void infer(std::vector<cv::Mat> img_batch, std::vector<std::vector<Detection>>& res_batch);
    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;
    Logger gLogger;
    int kOutputSize;

    // Deserialize the engine from file
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    
    float* device_buffers[2];
    float *output_buffer_host;
    float *decode_ptr_host;
    float *decode_ptr_device;

    int model_bboxes;        
    cudaStream_t stream;
};

#endif // YOLOV8MODEL_H
