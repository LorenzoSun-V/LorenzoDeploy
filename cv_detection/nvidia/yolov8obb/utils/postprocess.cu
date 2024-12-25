/*
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yolov8obb/utils/postprocess.cu
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-24 16:58:08
 */

#include "postprocess.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

// Device function to compute covariance matrix elements for a bounding box
__device__ void covariance_matrix(const BBox& box, float& a_val, float& b_val, float& c_val) {
    float w = box.w;
    float h = box.h;
    float rad = box.radian;
    
    float a = (w * w) / 12.0f;
    float b = (h * h) / 12.0f;
    float c = rad;
    
    float cos_r = cosf(c);
    float sin_r = sinf(c);
    
    float cos_r2 = cos_r * cos_r;
    float sin_r2 = sin_r * sin_r;
    
    a_val = a * cos_r2 + b * sin_r2;
    b_val = a * sin_r2 + b * cos_r2;
    c_val = (a - b) * cos_r * sin_r;
}

// Device function to compute probiou between two bounding boxes
__device__ float compute_probiou(const BBox& res1, const BBox& res2, float eps = 1e-7f) {
    float a1, b1, c1;
    float a2, b2, c2;
    
    // Compute covariance matrices for both boxes
    covariance_matrix(res1, a1, b1, c1);
    covariance_matrix(res2, a2, b2, c2);
    
    // Compute distance components
    float dx = res1.center_x - res2.center_x;
    float dy = res1.center_y - res2.center_y;
    
    // Compute terms t1, t2, t3 based on covariance matrices and positions
    float numerator_t1 = (a1 + a2) * dy * dy + (b1 + b2) * dx * dx;
    float denominator = (a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2) + eps;
    float t1 = numerator_t1 / denominator;
    
    float t2 = ((c1 + c2) * dx * dy) / denominator;
    
    float t3_numerator = (a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2);
    float t3_denominator = 4.0f * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps;
    float t3 = logf((t3_numerator / t3_denominator) + eps);
    
    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = fmaxf(fminf(bd, 100.0f), eps);
    float hd = sqrtf(1.0f - expf(-bd) + eps);
    
    return 1.0f - hd;
}

// CUDA Kernel for decoding OBB predictions
__global__ void decode_obb_kernel(
    const float* predict,            // Raw predictions [num_bboxes * bbox_element]
    DecodedBBox* decoded_boxes,      // Output decoded boxes [max_objects]
    int num_bboxes,
    int num_classes,
    float confidence_threshold,
    int max_objects
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bboxes) return;
    
    // Each prediction has (5 + num_classes) elements: cx, cy, w, h, radian, classes
    const float* box_ptr = predict + idx * (5 + num_classes);
    
    // Decode bounding box parameters
    BBox box;
    box.center_x = box_ptr[0];
    box.center_y = box_ptr[1];
    box.w = box_ptr[2];
    box.h = box_ptr[3];
    box.radian = box_ptr[4];
    
    // Find the class with the highest confidence
    float max_conf = -1.0f;
    int class_id = -1;
    for(int c = 0; c < num_classes; ++c){
        float class_conf = box_ptr[5 + c];
        if(class_conf > max_conf){
            max_conf = class_conf;
            class_id = c;
        }
    }
    
    // Final confidence score
    float confidence = max_conf; // Assuming objectness is part of class_conf
    
    // Apply confidence threshold
    if(confidence >= confidence_threshold && class_id != -1){
        // Write to decoded_boxes if within max_objects
        if(idx < max_objects){
            decoded_boxes[idx].center_x = box.center_x;
            decoded_boxes[idx].center_y = box.center_y;
            decoded_boxes[idx].w = box.w;
            decoded_boxes[idx].h = box.h;
            decoded_boxes[idx].radian = box.radian;
            decoded_boxes[idx].confidence = confidence;
            decoded_boxes[idx].class_id = class_id;
        }
    }
    else{
        // Mark as invalid by setting confidence to 0
        if(idx < max_objects){
            decoded_boxes[idx].confidence = 0.0f;
        }
    }
}

// CUDA Kernel for NMS of OBBs
__global__ void nms_obb_kernel(
    DecodedBBox* decoded_boxes, // [max_objects]
    int num_boxes,
    float iou_threshold
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_boxes) return;
    
    // Early exit if this box is already suppressed
    if(decoded_boxes[idx].confidence == 0.0f) return;
    
    // Compare with other boxes
    for(int j = idx + 1; j < num_boxes; ++j){
        if(decoded_boxes[j].confidence == 0.0f) continue;
        
        // Only compare boxes of the same class
        if(decoded_boxes[j].class_id != decoded_boxes[idx].class_id) continue;
        
        // Compute probiou
        BBox box1 = {
            decoded_boxes[idx].center_x,
            decoded_boxes[idx].center_y,
            decoded_boxes[idx].w,
            decoded_boxes[idx].h,
            decoded_boxes[idx].radian,
            decoded_boxes[idx].confidence,
            decoded_boxes[idx].class_id
        };
        BBox box2 = {
            decoded_boxes[j].center_x,
            decoded_boxes[j].center_y,
            decoded_boxes[j].w,
            decoded_boxes[j].h,
            decoded_boxes[j].radian,
            decoded_boxes[j].confidence,
            decoded_boxes[j].class_id
        };
        float iou = compute_probiou(box1, box2);
        
        if(iou >= iou_threshold){
            // Suppress box j by setting its confidence to 0
            decoded_boxes[j].confidence = 0.0f;
        }
    }
}

// Host function to decode OBB on GPU
void cuda_decode_obb(
    const float* predict,
    int num_bboxes,
    int num_classes,
    float confidence_threshold,
    DecodedBBox* parray,
    int max_objects,
    cudaStream_t stream
){
    // Define CUDA kernel launch parameters
    int threads = 256;
    int blocks = (num_bboxes + threads - 1) / threads;
    
    // Launch the decode kernel
    decode_obb_kernel<<<blocks, threads, 0, stream>>>(
        predict,
        parray,
        num_bboxes,
        num_classes,
        confidence_threshold,
        max_objects
    );
    
    // Error checking
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to launch decode_obb_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Host function to perform NMS on GPU
void cuda_nms_obb(
    DecodedBBox* parray,
    float nms_threshold,
    int max_objects,
    cudaStream_t stream
){
    // Define CUDA kernel launch parameters
    int threads = 256;
    int blocks = (max_objects + threads - 1) / threads;
    
    // Launch the NMS kernel
    nms_obb_kernel<<<blocks, threads, 0, stream>>>(
        parray,
        max_objects,
        nms_threshold
    );
    
    // Error checking
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to launch nms_obb_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Host function to decode and perform NMS on GPU
void cuda_decode_and_nms_obb(
    const float* predict,
    int num_bboxes,
    int num_classes,
    float confidence_threshold,
    float nms_threshold,
    DecodedBBox* parray,
    int max_objects,
    cudaStream_t stream
){
    // First, decode the bounding boxes
    cuda_decode_obb(
        predict,
        num_bboxes,
        num_classes,
        confidence_threshold,
        parray,
        max_objects,
        stream
    );
    
    // Wait for decoding to finish before starting NMS
    cudaStreamSynchronize(stream);
    
    // Then, perform NMS
    cuda_nms_obb(
        parray,
        nms_threshold,
        max_objects,
        stream
    );
    
    // Note: Synchronization is handled by the caller after this function
}
