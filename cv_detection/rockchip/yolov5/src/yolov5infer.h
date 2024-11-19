/*
 * @FilePath: /bt_alg_api/rkinfer/src/rkinfer.h
 * @Description: 瑞星微模型推理接口
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 18:07:06
 */
#pragma once

#include "common.h"
#include "opencv2/opencv.hpp"

extern "C"
{
    /*
     * @brief 初始化获得句柄
     *
	 * @param  pWeightfile          获得句柄
	 * @param  classnum             类别数量
     *
     * @return pDeepInstance        返回句柄
     */    
    ENUM_ERROR_CODE InitRKInferenceInstance(
        const char* pWeightfile, 
        int classnum, 
        void** pDeepInstance
    );
    
   	/*
     * @brief 输出模型检测结果
     *	     
     * @param  pDeepInstance         输入模型句柄
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
    * @brief 销毁句柄
    *
    * @param   pDeepInstance         需要销毁的句柄 
    *
    * @return  ENUM_ERROR_CODE       返回0表示成功
    */

    ENUM_ERROR_CODE DestoryInferenceInstance(
        void** pDeepInstance
    );
}
