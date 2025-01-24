/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 09:56:53
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-10-17 13:08:34
 * @Description:  YOLOv10模型前处理、推理、后处理代码
 */
#ifndef YOLOV10MODELMANAGER_H
#define YOLOV10MODELMANAGER_H

#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "common.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;


class YOLOV10ModelManager {
public:
    YOLOV10ModelManager();
    ~YOLOV10ModelManager();
    bool loadModel(const std::string model_path);
    bool inference(cv::Mat frame, std::vector<DetBox>& detBoxs);
    // bool batchInference(std::vector<cv::Mat> img_frames, std::vector<std::vector<DetBox>>& batchDetBoxes);

private:
    const int kOutputSize = 300;
    float factor;      // the factor(calculate in the preProcess) is used to adjust bbox size in the postProcess
    float conf = 0.25;
    int m_kBatchSize;
    int m_channel;
    int m_kInputH;
    int m_kInputW;
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node> input_port;
    ov::Tensor input_tensor;
    std::vector<float> input_data;
    float* output_data;

    void preprocess(cv::Mat* img, std::vector<float>& data);
    bool postProcess(float* output_data, std::vector<DetBox>& detBoxs);
    bool doInference(cv::Mat img, std::vector<DetBox>& detBoxs);
};

#endif // YOLOV10MODEL_H
