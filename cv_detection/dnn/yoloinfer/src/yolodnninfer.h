/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-23 08:59:42
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-23 15:16:03
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
     * @param  pDNNInstance          输出模型句柄
     *
     * @return ENUM_ERROR_CODE        返回错误码
     */
     
    ENUM_ERROR_CODE LoadDNNModelModules(
        const char* pWeightsfile,
        void** pDNNInstance
    );                                 
         
    /*
     * @brief 输出模型检测结果
     *         
     * @param  pDNNInstance         传入模型句柄
     * @param  frame                 输入检测图片  
     * @param  detBoxs               输出检测框
     *
     * @return  ENUM_ERROR_CODE      返回错误码
     */
    ENUM_ERROR_CODE InferenceDNNGetDetectResult(
        void* pDNNInstance,
        cv::Mat frame,
        std::vector<DetBox> &detBoxs
    );   

    /*
    * @brief 销毁句柄
    *
    * @param pDNNInstance          需要销毁的句柄
    *
    * @return  ENUM_ERROR_CODE      返回0表示成功
    */

    ENUM_ERROR_CODE DestoryDNNModeInstance(
        void** pDNNInstance
    );
               
}
