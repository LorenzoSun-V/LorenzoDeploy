#ifndef YOLOE2EV2MODELMANAGER_H
#define YOLOE2EV2MODELMANAGER_H

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


class YOLOE2Ev2ModelManager {
public:
    YOLOE2Ev2ModelManager();
    ~YOLOE2Ev2ModelManager();
    bool loadModel(const std::string engine_name);
    bool deserializeEngine(const std::string engine_name);
    bool inference(cv::Mat frame, std::vector<DetBox>& detBoxs);

private:
    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;

    void preprocess(cv::Mat& img, std::vector<float>& data);
    void rescale_box(std::vector<DetBox>& pred_box, std::vector<DetBox>& detBoxs, int width, int height);
    // Deserialize the engine from file
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;

    std::vector<float> input;
    int output0[1];
    float output1[1 * 200 * 4];
    float output2[1 * 200];
    float output3[1 * 200];
    void* buffers[5];

};

#endif // YOLOE2EV2MODEL_H
