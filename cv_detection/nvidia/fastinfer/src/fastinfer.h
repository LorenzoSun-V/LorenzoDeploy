/**
* @file   fastinfer.h.
*
* @brief
*
* 深度学习模型接口函数
*
* @copyright
*
* All Rights Reserved.
*/

#pragma once

#include "common.h"
#include <opencv2/opencv.hpp>


extern "C"
{
        
    struct BatchDetBox {
        DetBox *pdetbox;
        int ndetcount;//返回单张图片目标数量
    };     

    struct BatchFrames {
        cv::Mat frame;
    }; 

    //配置模型适配类型   
    typedef enum {
        MODEL_CODE_0=0,
        MODEL_CODE_1=1,
        MODEL_CODE_2=2,
        MODEL_CODE_3=3,
        MODEL_CODE_4=4,
        MODEL_CODE_5=5,
        MODEL_CODE_6=6,
        MODEL_CODE_7=7,
        MODEL_CODE_8=8,
        MODEL_CODE_9=9,
    }MODEL_TYPE_E;

    typedef enum {
        RUN_CPU=0,
        RUN_GPU=1,
        RUN_TRT=2,
    }RUN_PLATFROM_E;

	/*
     * @brief 图片处理接口
     *
	 * @param  pImagePath           输入图片地址 
	 * @param  frame                返回图片frame
     *
     * @return  ENUM_ERROR_CODE     返回错误码
     */    
    ENUM_ERROR_CODE ReadFrameFromPath(
        const char* pImagePath, 
        cv::Mat& frame
        );
                   

	/*
     * @brief 模型初始化函数
     *
	 * @param  pModelPath             输入模型路径，0~2输入文件，其他输入目录
	 * @param  pSerializefile         输入序列化的文件，加快第一次推理速度，ePlatfrom=RUN_TRT有效，其它方式为NULL
	 * @param  iStartnumber           输入数字，在模型输出类别ID基础上加上目前数字
	 * @param  eModelVersion          输入配置模型版本
     * @param  ePlatfrom              输入配置运行平台
     *
     * @return void*                  返回句柄指针
     */
	void* LoadDeepModelModules(
        const char* pModelpath,
        const char* pSerializefile,
		int iStartnumber,  
        MODEL_TYPE_E eModelversion,
        RUN_PLATFROM_E  ePlatfrom        
    );                                 
            
  	/*
     * @brief 输出模型检测结果
     *	     
     * @param  pDeepInstance         传入模型句柄
     * @param  frame                 输入检测图片
	 * @param  pDetBoxs              返回检测框坐标置信度ID信息
	 * @param  pndetcount            返回检测框数据量
     *
     * @return  ENUM_ERROR_CODE      返回错误码
     */
	ENUM_ERROR_CODE InferenceGetDetectResult(
		void* pDeepInstance,
	    cv::Mat frame,
	    DetBox* pDetBoxs,
	    int* pndetcount
    );   
  
     /*
     * @brief 批量输出模型检测结果（参照具体使用流程）
     *	     
     * @param  pDeepInstance         传入模型句柄
     * @param  pBatchframes          输入批量检测图片
     * @param  framecount            输入批量检测图片数量
	 * @param  pBatchDetBoxs         批量返回检测结果
     *
     * @return  ENUM_ERROR_CODE      返回错误码
     */
	ENUM_ERROR_CODE BatchInferenceGetDetectResult(
		void* pDeepInstance,
	    BatchFrames* pBatchframes,
	    int framecount,
	    BatchDetBox* pBatchDetBoxs
    ); 
              
     /*
     * @brief 标记识别框，返回带框图像
     *	   
     * @param   frame                   输入检测图片/返回结果 
     * @param   pDetBoxs                输出检测框  
     * @param   detCount                输出检测框个数          
     * 
     * @return  ENUM_ERROR_CODE         返回错误码
     */
	ENUM_ERROR_CODE RectDetectResultForImage(
	    cv::Mat &frame,
        DetBox* pDetBoxs, 
        int detCount
    );                                      
   
     /*
      * @brief 输出当前时间戳
      * @return  double         毫秒级
      */
    double GetCurrentTimeStampMS();    
          
   /*
    * @brief 销毁句柄
    *
    * @param instance               需要销毁的句柄
    *
    * @return  ENUM_ERROR_CODE      返回0表示成功
    */

    ENUM_ERROR_CODE DestoryDeepmodeInstance(
        void** pDLInstance
    );
               
}
