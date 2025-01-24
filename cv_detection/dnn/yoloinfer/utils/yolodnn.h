/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-23 08:58:59
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-23 15:36:49
 * @Description: 
 */
#ifndef YOLODNNMODELMANAGER_H
#define YOLODNNMODELMANAGER_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "common.h"

// YOLO模型参数结构体
typedef struct {
    int input_channel;  // 输入通道数
    int input_height;   // 输入高度
    int input_width;    // 输入宽度
    int batch_size;     // 批处理大小
    float conf_thresh;  // 置信度阈值
    float iou_thresh;   // IOU阈值
    float x_factor;     // X轴缩放因子
    float y_factor;     // Y轴缩放因子
    bool yolov8 = false; // 是否为YOLOv8模型
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
} model_param_t;

// YOLO DNN模型管理类
// 负责YOLO模型的加载、推理和后处理
class YOLODNNModelManager {
public:
    YOLODNNModelManager();  // 构造函数
    ~YOLODNNModelManager(); // 析构函数
    
    // 加载YOLO模型
    // @param model_name: 模型文件路径
    // @return: 成功返回true，失败返回false
    bool loadModel(const std::string model_name);
    
    // 执行推理
    // @param frame: 输入图像
    // @param detBoxs: 输出检测框
    // @return: 成功返回true，失败返回false
    bool inference(cv::Mat& frame, std::vector<DetBox>& detBoxs);

private:
    model_param_t m_model_param;  // 模型参数
    cv::dnn::Net net;             // OpenCV DNN网络
    std::vector<cv::Mat> outputs; // 网络输出
    
    // 图像预处理
    // @param img: 输入图像
    // @return: 预处理后的图像
    cv::Mat preprocess(cv::Mat img);
    
    // 执行推理
    // @param modelInput: 模型输入
    void doInference(cv::Mat modelInput);
    
    // 后处理
    // @param detBoxs: 输出检测框
    void postprocess(std::vector<DetBox>& detBoxs);
};

#endif // YOLODNNMODELMANAGER_H
