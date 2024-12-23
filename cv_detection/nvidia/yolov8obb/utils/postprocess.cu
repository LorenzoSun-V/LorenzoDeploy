/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-12 14:49:46
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-13 11:52:54
 * @Description: YOLOv8 OBB CUDA后处理
 */
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>


#include "postprocess.h"


// GPU上计算概率IoU的 device 函数
static __device__ void convariance_matrix(float w, float h, float r, float& a, float& b, float& c){
    float a_val = w * w / 12.0f;
    float b_val = h * h / 12.0f;
    float cos_r = cosf(r); 
    float sin_r = sinf(r);

    a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
    b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
    c = (a_val - b_val) * sin_r * cos_r;
}

static __device__ float box_probiou(
    float cx1, float cy1, float w1, float h1, float r1,
    float cx2, float cy2, float w2, float h2, float r2,
    float eps = 1e-7
){

    // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    float a1, b1, c1, a2, b2, c2;
    convariance_matrix(w1, h1, r1, a1, b1, c1);
    convariance_matrix(w2, h2, r2, a2, b2, c2);

    float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) / (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) + eps); 
    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = fmaxf(fminf(bd, 100.0f), eps);
    float hd = sqrtf(1.0f - expf(-bd) + eps);
    return 1 - hd;    
}

static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects){  
    // cx, cy, w, h, cls, angle

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem            = predict + (5 + num_classes) * position;
    float* class_confidence = pitem + 4;
    float confidence        = *class_confidence++;
    int label               = 0;
    // 打印每个线程的confidence值
    printf("Thread [%d, %d, %d]: Position %d, Confidence: %f, Label: %d\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           position, confidence, label);
    // 通过循环找到置信度最高的类别，并记录对应的 label。
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }

    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;

    float cx         = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;
    float angle      = *(pitem + num_classes);

    float* pout_item = parray + 1 + index * 8;
    *pout_item++ = cx;
    *pout_item++ = cy;
    *pout_item++ = width;
    *pout_item++ = height;
    *pout_item++ = angle;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;
        
        // cx, cy, w, h, angle, confidence, class_label, keepflag
        float* pcurrent = bboxes + 1 + position * 8;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * 8;
            if(i == position || pcurrent[6] != pitem[6]) continue;

            if(pitem[5] >= pcurrent[5]){
                if(pitem[5] == pcurrent[5] && i < position)
                    continue;

                float iou = box_probiou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pcurrent[4],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3],    pitem[4]
                );

                if(iou > threshold){
                    pcurrent[7] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    } 

void cuda_decode_obb(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects, cudaStream_t stream) {
    int block = 256;
    int grid = ceil(num_bboxes / (float)block);
    decode_kernel<<<grid, block, 0, stream>>>((float*)predict, num_bboxes, num_classes, confidence_threshold, parray, max_objects);
}

void cuda_nms_obb(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}