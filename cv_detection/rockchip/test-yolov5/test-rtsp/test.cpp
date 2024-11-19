/*
 * @FilePath: /bt_alg_api/test-rkinfer/test-rtsp/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 18:06:28
 */
/**
* @file      test.cpp
*
* @brief     单batch测试代码
*
* @copyright 无锡宝通智能科技股份有限公司
*
* @author  图像算法组-贾俊杰
*
* All Rights Reserved.
*/
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "common.h"
#include "rkinfer.h"
#include "videocodec.h"
#include "utils.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 3){
		cout<<"param1: the make binary;"<<endl;
		cout<<"param2: model path is the model and prototxt path;"<<endl;
		cout<<"param3: input the detect image file;"<<endl;
		cout<<"example: ./binary imagepath weightsfile classfile pChinesefile "<<endl;
		exit(-1);
 	}

    const char* pWeightsfile = argv[1];
    const char* pInputRtspUrl = argv[2];
 	//初始化句柄	    
    void * pRkInferInstance = NULL;
    ENUM_ERROR_CODE code = InitRKInferenceInstance(pWeightsfile, 2, &pRkInferInstance);
    if(pRkInferInstance == NULL || code != ENUM_OK){
        cout<<"can not get pRkInferInstance!"<<endl;
        return -1;
    }

    void * pVideostreamInstance = NULL;
    code = InitVideoStreamDecoderInstance(pInputRtspUrl,ROCKCHIP_3399PRO_H264, &pVideostreamInstance);
    if(code != ENUM_OK || NULL == pVideostreamInstance)
    {
        cout<<"can not get pDecoderInstance!"<<endl;
        return -1;
    }    

    cout<<"Init Finshed!"<<endl;  

    int count = 0;
    cv::Mat frame;
    while (true)
    {
        cout<<"GetOneFrameFromDecoderInstance"<<endl;
        code = GetOneFrameFromDecoderInstance(pVideostreamInstance, frame);
        if(code != ENUM_OK){
            cout<<"Can not read video data!"<<endl;
            break;
        }
        std::vector<DetBox> detResult;
        //推理获得检测的目标及数量
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pRkInferInstance, frame, detResult);
        double t_detect_end = GetCurrentTimeStampMS();  
            

        //将结果画到图上再保存下来
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);    
        std::cout<<"size: "<<detResult.size()<<std::endl;
        DrawRectDetectResultForImage(frame, detResult);  	    

        if(detResult.size() > 0 )
        {
            count++;
            char imagename[100];
            sprintf(imagename, "result_%d.jpg",count);
            cv::imwrite(imagename, frame);
            if(count == 100) break;
        }
    }
    DestoryRtspStreamDecoderInstance(&pVideostreamInstance);
    DestoryInferenceInstance(&pRkInferInstance);           
    std::cout << "Finish !"<<endl;
    return 0;
}


