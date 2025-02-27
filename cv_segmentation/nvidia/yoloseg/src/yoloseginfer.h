/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-26 08:51:35
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-06 17:41:05
 * @Description: 
 */
#pragma once

#include "common.h"
#include <opencv2/opencv.hpp>


extern "C"
{

    /*
     * @brief 模型初始化函数
     *
     * @param  pWeightsfile           输入模型权重文件
     * @param  kBatchSize             模型图片推理数量
     * @param  pDeepInstance          返回输入视句柄指针
     * @param  bUseYOLOv8             是否使用YOLOv8模型
     * @return ENUM_ERROR_CODE        返回错误码     
     */
     
    ENUM_ERROR_CODE LoadInstanceSegmentModelModules(
        const char* pWeightsfile,
        void** pDeepInstance,
        bool bUseYOLOv8 = true
    );                                 
         
    /*
     * @brief  输出模型检测结果
     * @param  pDeepInstance         传入模型句柄
     * @param  frame                 输入检测图片  
     * @param  segBoxs               返回检测框
     * @param  masks                 返回mask结果
     * @return ENUM_ERROR_CODE       返回错误码
     */
    ENUM_ERROR_CODE InferenceGetInstanceSegmentResult(
        void* pDeepInstance,
        cv::Mat frame,
        std::vector<SegBox>& segBoxs,
        std::vector<cv::Mat>& masks
    );   
                                           
   
    /*
     * @brief  批量输出模型检测结果
     * @brief  多batch推理每次,送入图像需小于等于模型设置batch数量            
     * @param  pDeepInstance         传入模型句柄
     * @param  batchframes           批量输入批量检测图片，最大数量根据模型batch决定
     * @param  batchsegBoxs          批量输出检测结果
     * @param  batchMasks            批量输出mask结果
     * @return ENUM_ERROR_CODE       返回错误码
     */
    ENUM_ERROR_CODE BatchInferenceGetInstanceSegmentResult(
        void* pDeepInstance,
        std::vector<cv::Mat> batchframes,
        std::vector<std::vector<SegBox>>& batchsegBoxs,
        std::vector<std::vector<cv::Mat>>& batchMasks
    );
     
   /*
    * @brief 销毁句柄
    *
    * @param instance               需要销毁的句柄
    *
    * @return  ENUM_ERROR_CODE      返回0表示成功
    */

    ENUM_ERROR_CODE DestoryDeepmodeInstance(
        void** pDeepInstance
    );
               
}
