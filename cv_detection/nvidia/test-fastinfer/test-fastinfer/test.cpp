/*
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/test-fastinfer/test-fastinfer/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-16 15:19:33
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "fastinfer.h"

int main(int argc, char* argv[])
{
	if(argc < 3){
		std::cout<<"param1: the make binary;"<<std::endl;
		std::cout<<"param2: model path is the model and prototxt path;"<<std::endl;
		std::cout<<"param3: input the detect image file;"<<std::endl;
		std::cout<<"example: ./binary video param weights"<<std::endl;
		exit(-1);
	}

	const char* pImagefile = argv[1];
	const char* pWeightsfile = argv[2];
	const char* pSerializefile = argv[3];

 	cv::Mat frame;   	    
	void * pDNNInstance = LoadDeepModelModules(pWeightsfile, pSerializefile, 1, MODEL_CODE_9, RUN_TRT);
    if(pDNNInstance == NULL ){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    }
    
 	DetBox *pDetBox = new DetBox[100];
    if (NULL == pDetBox) {
        std::cout<<"pDetBox is NULL!"<<std::endl;
        return -1;
    }
    int detCount = 0;
       
    std::cout<<"Init Finshed!"<<std::endl;  
    ReadFrameFromPath(pImagefile, frame);
    for(int i=0;i<100;i++) {
	    InferenceGetDetectResult(pDNNInstance, frame, pDetBox, &detCount);
    }

    double diff_time=0.0, total_time=0.0;
    for(int i=0;i<1000;i++)
    {
		
	    memset(pDetBox, 0, sizeof(DetBox)*100);
	    double t_detect_start = GetCurrentTimeStampMS();
	    InferenceGetDetectResult(pDNNInstance, frame, pDetBox, &detCount);
	    double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
	    diff_time = t_detect_end - t_detect_start;
        total_time += diff_time;
	    //RectDetectResultForImage(frame, pDetBox, detCount);    
        //cv::imwrite("image.jpg",frame);
		//cv::imshow("frame",frame);
        //waitKey(1);
        //break;
    }

    if (total_time > 0) {
        std::cout << "total_time= " << total_time 
                  << " mean fps= "   <<1000/(total_time/1000) << std::endl;  
    }
    
    DestoryDeepmodeInstance(&pDNNInstance);           
    std::cout << "Finish !"<<std::endl;
    return 0;
}


