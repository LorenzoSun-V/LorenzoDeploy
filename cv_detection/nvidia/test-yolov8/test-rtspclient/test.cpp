/*
 * @FilePath: /bt_alg_api/test-yolov8/test-rtspclient/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 16:56:54
 */
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include "yolov8infer.h"
#include "videocodec.h"
#include "utils.h"
using namespace cv;
using namespace std;


int main(int argc, char* argv[]) 
{
	if(argc < 4){
        cout<<"param1: the make binary;"<<endl;
        cout<<"param2: model path is the model and prototxt path;"<<endl;
        cout<<"param3: input the detect image file;"<<endl;
        cout<<"example: ./binary weightsfile imagepath"<<endl;
        exit(-1);
 	}
    
    const char* pWeightsfile = argv[1];
    const char* pInputRtspUrl = argv[2];
    int save_seconds = atoi(argv[3]);

	//初始化RTSP流
    void * pVideostreamInstance = NULL;
	ENUM_ERROR_CODE eOK = InitVideoStreamDecoderInstance(pInputRtspUrl, NVIDIA_JETSON_H264, &pVideostreamInstance);
	if(eOK != ENUM_OK || NULL == pVideostreamInstance)
	{
        cout<<"can not get pDecoderInstance!"<<endl;
        return -1;
    } 

	//保存本地视频
 	int frame_fps=25;
	void* pSaveVideoInstance = NULL;
	eOK = InitVideoStreamInstance("test.mp4", NVIDIA_JETSON_H264, frame_fps, 320, &pSaveVideoInstance);
	if(pSaveVideoInstance == NULL ) {
	    cout<<"can not init pSaveVideoInstance!"<<endl;
	    return -1;
	}

    void * pDNNInstance= NULL; 
    eOK =  LoadDeepModelModules(pWeightsfile, 1, &pDNNInstance);
    if(eOK != ENUM_OK && NULL == pDNNInstance){
        cout<<"can not get pDNNInstance!"<<endl;
        return -1;
    }

    int framenum = 0; 
    cv::Mat frame;
    while(true)
    {
         //获得解码MAT帧
        eOK = GetOneFrameFromDecoderInstance(pVideostreamInstance, frame);
        if(eOK != ENUM_OK){
        	cout<<"Can not read video data!"<<endl;
        	break;
        }
  
        std::vector<DetBox> detResult;
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pDNNInstance, frame, detResult);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start); 
           
        DrawRectDetectResultForImage(frame, detResult);  	 
    	
         //将MAT帧进行存储
        eOK = WriteOneFrameToInstance(pSaveVideoInstance, frame);
        if(eOK == ENUM_OK)
        {
	        if(framenum == frame_fps * save_seconds ) 
	        {
                cout<<"Complete save video!"<<endl;	
                DestorySaveVideoInstance(pSaveVideoInstance);
                break;
	        }	
        	framenum++;
        }	   
    }

    DestoryRtspStreamDecoderInstance(&pVideostreamInstance);
    DestoryDeepmodeInstance(&pDNNInstance);           
    std::cout << "Finish !"<<endl;
    return 0;
}
