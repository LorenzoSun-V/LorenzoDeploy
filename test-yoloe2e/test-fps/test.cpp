/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-09-23 10:55:43
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-10-17 13:35:38
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "utils.h"
#include "common.h"
#ifdef E2E_V1
    #include "yoloe2einfer.h"
#else
    #include "yoloe2ev2infer.h"
#endif


int main(int argc, char* argv[])
{
    if(argc < 2) {
        std::cout<<"example: ./binary image_path .engine"<<std::endl;
        exit(-1);
    }

    const char* pImagefile = argv[1];
    const char* pWeightsfile = argv[2];

    if(pImagefile == NULL || pWeightsfile == NULL) {
        std::cout<<"input param error!"<<std::endl;
        return -1;
    }

    void * pDeepInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDeepInstance);
    if(eOK != ENUM_OK) {
        std::cout<<"can not get pDeepInstance!"<<std::endl;
        return -1;
    } 
    
    int detCount = 0;
       
    std::cout<<"Init Finshed!"<<std::endl; 

    cv::Mat frame;        
    ReadFrameFromPath(pImagefile, frame);
    std::vector<DetBox> pDetBox;
    for(int i=0;i<100;i++) {
        InferenceGetDetectResult(pDeepInstance, frame, pDetBox);
    }

    double diff_time=0.0, total_time=0.0;
    for(int i=0;i<1000;i++)
    {
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pDeepInstance, frame, pDetBox);
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
    DestoryDeepmodeInstance(&pDeepInstance);           
    std::cout << "Finish !"<<std::endl;
    return 0;
}


