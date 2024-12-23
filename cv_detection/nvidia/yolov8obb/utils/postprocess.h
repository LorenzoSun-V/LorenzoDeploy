/*
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yolov8obb/utils/postprocess.h
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-13 10:05:44
 */
#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "common.h"
#include "NvInfer.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <tuple> 

struct BBox {
    float center_x, center_y, w, h, radian; // 中心点坐标 (x, y), 宽度 w, 高度 h, 弧度
    float score;             // 置信度分数
    int class_id;            // 类别 ID
};

typedef struct {
    int input_channel;
    int input_width;
    int input_height;
    int num_classes;
    int batch_size;
    int num_bboxes;
    int bbox_element;  // center_x, center_y, w, h, n*cls, obj
    int max_objects = 1000;
    float conf_thresh = 0.25f;
    float iou_thresh = 0.45f;
} model_param_t;

// CPU NMS
void nms_obb_batch(std::vector<std::vector<BBox>>& batch_res, float* output,  model_param_t model_param);
// GPU NMS
void cuda_decode_obb(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects, cudaStream_t stream);
void cuda_nms_obb(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);
//解码转换结果到输出结构体
bool postprocess_batch(std::vector<std::vector<BBox>> batch_bboxes, std::vector<cv::Mat> images, 
                 int model_width, int model_height, std::vector<std::vector<DetBox>>& batch_res);
#endif 