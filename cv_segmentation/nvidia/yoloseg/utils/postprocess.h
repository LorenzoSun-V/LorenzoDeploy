/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-27 14:28:30
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-06 17:40:16
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

/*
    * @brief  对一批检测框进行非极大值抑制 (NMS)
    * @param  batch_res             返回批次的检测框结果，每个元素对应一张图像的检测框
    * @param  output                模型推理输出，包含所有检测框的数据
    * @param  model_param           模型参数，包括输入输出尺寸、类别数等信息
    * @param  m_buseyolov8          是否使用 YOLOv8 模型（true: YOLOv8, false: YOLOv5）
    * @return bool                  返回是否成功执行
*/
bool batch_nms(
    std::vector<std::vector<InstanceSegResult>>& batch_res, 
    float* output, 
    model_param_t model_param, 
    bool m_buseyolov8);

/*
    * @brief  对一批检测框生成分割 mask
    * @param  proto                 模型的分割输出张量数据
    * @param  proto_size            分割输出张量的大小
    * @param  batch_dets            批次的检测框结果，每个元素对应一张图像的检测框
    * @param  batch_masks           返回批次的分割 mask 结果，每个元素对应一张图像的 mask
    * @param  model_param           模型参数，包括输入输出尺寸、类别数等信息
    * @return bool                  返回是否成功执行
*/
bool batch_process_mask(
    const float* proto, 
    int proto_size, 
    const std::vector<std::vector<InstanceSegResult>>& batch_dets, 
    std::vector<std::vector<cv::Mat>>& batch_masks, 
    const model_param_t& model_param);

/*
    * @brief  对检测框和分割 mask 进行后处理，转换为最终结果
    * @param  batch_bboxes          输入批次的检测框结果，每个元素对应一张图像的检测框
    * @param  batch_masks           输入批次的分割 mask 结果，每个元素对应一张图像的 mask
    * @param  batch_images          输入批次的原始图像
    * @param  model_width           模型输入的宽度
    * @param  model_height          模型输入的高度
    * @param  batch_bboxes_result   返回批次的最终检测框结果，每个元素对应一张图像的检测框
    * @param  batch_masks_result    返回批次的最终分割 mask 结果，每个元素对应一张图像的 mask
    * @return bool                  返回是否成功执行
*/
bool postprocess_batch(
    const std::vector<std::vector<InstanceSegResult>>& batch_bboxes,
    const std::vector<std::vector<cv::Mat>>& batch_masks,
    const std::vector<cv::Mat>& batch_images, 
    int model_width, 
    int model_height, 
    std::vector<std::vector<SegBox>>& batch_bboxes_result,
    std::vector<std::vector<cv::Mat>>& batch_masks_result);