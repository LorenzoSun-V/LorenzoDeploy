/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-27 14:28:30
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-06 09:10:58
 * @Description: 
 */
#pragma once

#include <opencv2/opencv.hpp>
#include "common.h"

struct InstanceSegResult {
    float center_x, center_y, w, h; // 中心点坐标 (x, y), 宽度 w, 高度 h
    float score;             // 置信度分数
    int class_id;            // 类别 ID
    float mask[32];          // mask系数
};

typedef struct {
    int batch_size;
    int input_channel;
    int input_width;
    int input_height;
    int num_classes;
    int num_bboxes;
    int bbox_element;
    int seg_output;
    int seg_output_height;
    int seg_output_width;
    float conf_thresh = 0.25f;
    float iou_thresh = 0.45f;
} model_param_t;

void batch_nms(
    std::vector<std::vector<InstanceSegResult>>& batch_res, 
    float* output, 
    model_param_t model_param, 
    bool yolov8);

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