/*
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yolov8obb/utils/postprocess.h
 * @Copyright: 
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-24 17:00:00
 */

#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "common.h"        // Assuming DetBox and other common definitions are here
#include "NvInfer.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <tuple>
#include <vector>
#include <map>

// Structure representing a bounding box (Oriented Bounding Box)
struct BBox {
    float center_x;  // X-coordinate of the center
    float center_y;  // Y-coordinate of the center
    float w;         // Width of the bounding box
    float h;         // Height of the bounding box
    float radian;    // Rotation angle in radians
    float score;     // Confidence score
    int class_id;    // Class ID
};

// Structure representing a decoded bounding box (for CUDA processing)
struct DecodedBBox {
    float center_x;
    float center_y;
    float w;
    float h;
    float radian;
    float confidence;
    int class_id;
};

// Structure holding model parameters
typedef struct {
    int input_channel;    // Number of input channels
    int input_width;      // Input image width
    int input_height;     // Input image height
    int num_classes;      // Number of object classes
    int batch_size;       // Batch size
    int num_bboxes;       // Number of bounding boxes per image
    int bbox_element;     // Number of elements per bounding box (e.g., 5 + num_classes)
    int max_objects = 1000; // Maximum number of objects to detect per image
    float conf_thresh = 0.25f; // Confidence threshold for filtering
    float iou_thresh = 0.45f;  // IoU threshold for NMS
} model_param_t;

/**
 * @brief Performs Non-Maximum Suppression (NMS) for a batch of images using CPU.
 * 
 * @param batch_res Output parameter to store the NMS results for each image.
 * @param output Raw output data from the model.
 * @param model_param Model parameters including thresholds and dimensions.
 */
void nms_obb_batch(std::vector<std::vector<BBox>>& batch_res, float* output, const model_param_t& model_param);

/**
 * @brief Decodes raw predictions from the model into structured bounding boxes on the GPU.
 * 
 * @param predict Pointer to the raw prediction data on the device.
 * @param num_bboxes Number of bounding boxes per image.
 * @param num_classes Number of object classes.
 * @param confidence_threshold Threshold to filter out low-confidence detections.
 * @param parray Pointer to the device memory where decoded bounding boxes will be stored.
 * @param max_objects Maximum number of objects to detect per image.
 * @param stream CUDA stream for asynchronous execution.
 */
void cuda_decode_obb(
    const float* predict,
    int num_bboxes,
    int num_classes,
    float confidence_threshold,
    DecodedBBox* parray,
    int max_objects,
    cudaStream_t stream
);

/**
 * @brief Performs Non-Maximum Suppression (NMS) on decoded bounding boxes on the GPU.
 * 
 * @param parray Pointer to the device memory containing decoded bounding boxes.
 * @param nms_threshold IoU threshold for suppressing overlapping boxes.
 * @param max_objects Maximum number of objects to keep per image after NMS.
 * @param stream CUDA stream for asynchronous execution.
 */
void cuda_nms_obb(
    DecodedBBox* parray,
    float nms_threshold,
    int max_objects,
    cudaStream_t stream
);

/**
 * @brief Combined function to decode predictions and perform NMS on the GPU.
 * 
 * @param predict Pointer to the raw prediction data on the device.
 * @param num_bboxes Number of bounding boxes per image.
 * @param num_classes Number of object classes.
 * @param confidence_threshold Threshold to filter out low-confidence detections.
 * @param nms_threshold IoU threshold for suppressing overlapping boxes.
 * @param parray Pointer to the device memory where decoded bounding boxes will be stored.
 * @param max_objects Maximum number of objects to detect per image.
 * @param stream CUDA stream for asynchronous execution.
 */
void cuda_decode_and_nms_obb(
    const float* predict,
    int num_bboxes,
    int num_classes,
    float confidence_threshold,
    float nms_threshold,
    DecodedBBox* parray,
    int max_objects,
    cudaStream_t stream
);

/**
 * @brief Post-processes the bounding boxes by transforming coordinates back to the original image space.
 * 
 * @param batch_bboxes Vector containing vectors of bounding boxes for each image after NMS.
 * @param images Vector of original input images.
 * @param model_width Width used during model preprocessing.
 * @param model_height Height used during model preprocessing.
 * @param batch_res Output parameter to store the final detection results for each image.
 * @return true if post-processing is successful.
 * @return false otherwise.
 */
bool postprocess_batch(
    const std::vector<std::vector<BBox>>& batch_bboxes,
    const std::vector<cv::Mat>& images, 
    int model_width, 
    int model_height, 
    std::vector<std::vector<DetBox>>& batch_res
);

#endif // POSTPROCESS_H
