/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-23 08:58:59
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-23 15:09:27
 * @Description: 
 */
#ifndef YOLODNNMODELMANAGER_H
#define YOLODNNMODELMANAGER_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "common.h"

typedef struct {
    int input_channel;
    int input_height;
    int input_width;
    int batch_size;
    float conf_thresh;
    float iou_thresh;
    float x_factor;
    float y_factor;
    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
} model_param_t;

class YOLODNNModelManager {
public:
    YOLODNNModelManager();
    ~YOLODNNModelManager();
    bool loadModel(const std::string model_name);
    bool inference(cv::Mat& frame, std::vector<DetBox>& detBoxs);

private:
    model_param_t m_model_param; 
    cv::dnn::Net net;
    std::vector<cv::Mat> outputs;
    cv::Mat preprocess(cv::Mat img);
    void doInference(cv::Mat modelInput);
    void postprocess(std::vector<DetBox>& detBoxs);
};

#endif // YOLODNNMODELMANAGER_H