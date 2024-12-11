/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 13:37:20
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-04 13:52:06
 * @Description: YOLOv8OBB模型GPU推理接口
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
     * @param  pDeepInstance          输出模型句柄
     *
     * @return ENUM_ERROR_CODE        返回错误码
     */
     
    ENUM_ERROR_CODE LoadDeepModelModules(
        const char* pWeightsfile,
        void** pDeepInstance
    );
         
    /*
     * @brief 输出模型检测结果
     *         
     * @param  pDeepInstance         传入模型句柄
     * @param  frame                 输入检测图片  
     * @param  detBoxs               输出检测框
     *
     * @return  ENUM_ERROR_CODE      返回错误码
     */
    ENUM_ERROR_CODE InferenceGetDetectResult(
        void* pDeepInstance,
        cv::Mat frame,
        std::vector<DetBox> &detBoxs
    );   

     /*
     * @brief 批量输出模型检测结果
     * @param  pDeepInstance         传入模型句柄
     * @param  batch_images          批量输入批量检测图片，最大数量根据模型batch决定
     * @param  batch_result          批量输出检测结果
     *
     * @return  ENUM_ERROR_CODE      返回错误码
     */
    ENUM_ERROR_CODE BatchInferenceGetDetectResult(
        void* pDeepInstance,
        std::vector<cv::Mat> batch_images,
        std::vector<std::vector<DetBox>> &batch_result
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
