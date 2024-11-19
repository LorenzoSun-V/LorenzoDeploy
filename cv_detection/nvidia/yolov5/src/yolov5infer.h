/*
 * @FilePath: /bt_alg_api/yolov5/src/yolov5infer.h
 * @Description: yolov5模型推理接口
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 17:02:37
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
     * @param  detBoxs               返回检测框
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
     * @brief  多batch推理每次,送入图像需小于等于模型设置batch数量            
     * @param  pDeepInstance         传入模型句柄
     * @param  batchframes           批量输入批量检测图片，最大数量根据模型batch决定
     * @param  batchDetBoxs          批量输出检测结果
     *
     * @return  ENUM_ERROR_CODE      返回错误码
     */
    ENUM_ERROR_CODE BatchInferenceGetDetectResult(
        void* pDeepInstance,
        std::vector<cv::Mat> batchframes,
        std::vector<std::vector<DetBox>> &batchDetBoxs
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
