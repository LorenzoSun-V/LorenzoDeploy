/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-27 14:28:30
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-06 11:41:24
 * @Description: 
 */
#pragma once

#include <opencv2/opencv.hpp>
#include "common.h"

struct InstanceSegResult {
    float center_x, center_y, w, h; // 中心点坐标 (x, y), 宽度 w, 高度 h
    float score;                   // 置信度分数
    int class_id;                  // 类别 ID
    float mask[32];               // mask系数
};

typedef struct {
    int batch_size;               // batch size
    int input_channel;            // 输入模型通道数
    int input_width;              // 输入模型宽度
    int input_height;             // 输入模型高度
    int num_classes;              // 类别数量
    int num_bboxes;               // 检测框数量
    int bbox_element;             // 检测框结构体元素数量
    int seg_output;               // 分割输出特征图通道数
    int seg_output_height;        // 分割输出特征图高度
    int seg_output_width;         // 分割输出特征图宽度
    float conf_thresh = 0.25f;    // 检测框置信度阈值
    float iou_thresh = 0.45f;     // NMS 阈值
} model_param_t;

void batch_nms(
    std::vector<std::vector<InstanceSegResult>>& batch_res, 
    float* output, 
    model_param_t model_param, 
    bool m_buseyolov8);

void batch_process_mask(
    const float* proto, 
    int proto_size, 
    const std::vector<std::vector<InstanceSegResult>>& batch_dets, 
    std::vector<std::vector<cv::Mat>>& batch_masks, 
    const model_param_t& model_param);

bool postprocess_batch(
    const std::vector<std::vector<InstanceSegResult>>& batch_bboxes,
    const std::vector<std::vector<cv::Mat>>& batch_masks,
    const std::vector<cv::Mat>& batch_images, 
    int model_width, 
    int model_height, 
    std::vector<std::vector<SegBox>>& batch_bboxes_result,
    std::vector<std::vector<cv::Mat>>& batch_masks_result);