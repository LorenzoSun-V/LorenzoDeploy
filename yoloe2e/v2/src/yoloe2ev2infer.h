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
