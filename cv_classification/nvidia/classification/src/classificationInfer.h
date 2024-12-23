/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 10:01:24
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-20 15:54:55
 * @Description: 特征提取推理接口
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
     * @param  objects               向量结果
     * @param  topK                  设置返回置信度最高的前多少个值，默认第一个
     *
     * @return  ENUM_ERROR_CODE      返回错误码
     */
    ENUM_ERROR_CODE InferenceGetFeature(
        void* pDeepInstance,
        cv::Mat frame,
        std::vector<ClsResult>& objects,
        int topK=1
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
