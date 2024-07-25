#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include "yolov8infer.h"
#include "videocodec.h"

using namespace cv;
using namespace std;


#define TEST_VIDEOSAVE 

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
    int savetime = atoi(argv[3]);

	//初始化视频解码
    void * pVideostreamInstance = NULL;
	ENUM_ERROR_CODE eOK = InitVideoStreamDecoderInstance(pInputRtspUrl, NVIDIA_JETSON_H264, &pVideostreamInstance);
	if(eOK != ENUM_OK || NULL == pVideostreamInstance)
	{
        cout<<"can not get pDecoderInstance!"<<endl;
        return -1;
    } 
#ifdef TEST_VIDEOSAVE 
	//保存本地视频
 	int frame_fps=60;
	void* pSaveVideoInstance = NULL;
	eOK = InitVideoStreamInstance("test.mp4", NVIDIA_JETSON_H264, frame_fps, 1024, &pSaveVideoInstance);
	if(pSaveVideoInstance == NULL ) {
	    cout<<"can not init pSaveVideoInstance!"<<endl;
	    return -1;
	}
	int framenum = 0;
#endif


    void * pDNNInstance= NULL; 
     eOK =  LoadDeepModelModules(pWeightsfile, 1, &pDNNInstance);
    if(eOK != ENUM_OK && NULL == pDNNInstance){
        cout<<"can not get pDNNInstance!"<<endl;
        return -1;
    } 
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

 #ifdef TEST_VIDEOSAVE          	
         //将MAT帧进行存储
        eOK = WriteOneFrameToInstance(pSaveVideoInstance, frame);
        if(eOK == ENUM_OK)
        {
	        if(framenum == frame_fps*savetime ) 
	        {
                cout<<"Complete save video!"<<endl;	
                DestorySaveVideoInstance(pSaveVideoInstance);
                break;
	        }	
        	framenum++;
        }	
#endif      

    }
    DestoryRtspStreamDecoderInstance( pVideostreamInstance);
    DestoryDeepmodeInstance(pDNNInstance);           
    std::cout << "Finish !"<<endl;
    return 0;
}
