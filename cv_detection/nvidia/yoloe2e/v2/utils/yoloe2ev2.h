/*
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/v2/utils/yoloe2ev2.h
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-07 16:08:16
 */
#ifndef YOLOE2EV2MODELMANAGER_H
#define YOLOE2EV2MODELMANAGER_H

#include "cuda.h"
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "logging.h"
#include "common.h"
#include <opencv2/opencv.hpp>

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
    bool batchinference(std::vector<cv::Mat> frames, std::vector<std::vector<DetBox>>& batchBoxes);

private:
    size_t  m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;
    int m_maxObject;
    void preprocess(cv::Mat& img, std::vector<float>& data);
    bool doInference();
    void rescale_box(std::vector<DetBox>& pred_box, std::vector<DetBox>& detBoxs, int width, int height);
    // Deserialize the engine from file
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;

    std::vector<float> input;
    int* host_output0;
    float* host_output1;
    float* host_output2;
    float* host_output3;
    void* buffers[5];

};

#endif // YOLOE2EV2MODEL_H
