/**
* @file    test.cpp
*
* @brief     串口接口单batch测试代码
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
#include "yolov5infer.h"
#include "opencv2/opencv.hpp"
#include "utils.h"

using namespace cv;
using namespace std;

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
	if(argc < 2) {
          std::cout<<"example: ./binary imagedir .bin"<<std::endl;
          exit(-1);
    }

    const char* pimagedir = argv[2];
    const char* pWeightsfile = argv[1];
          
    if(pimagedir == NULL || pWeightsfile == NULL)
    {
        std::cout<<"input param error!"<<std::endl;
        return -1;
    }
	
    std::vector<cv::Mat> batchframes; 
    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
  	cv::Mat frame; 
    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        batchframes.push_back(frame);
    }

    void * pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK && NULL == pDNNInstance){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 
    
    std::cout<<"Init Finshed!"<<std::endl;  
    int frame_num = static_cast<int>(batchframes.size());
    double total_time = 0.0;
    int index=1;
    std::vector<DetBox> detBoxs;
    for( auto& frame: batchframes)
    {
        detBoxs.clear();
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pDNNInstance, frame, detBoxs);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
        total_time += t_detect_end - t_detect_start;
        std::string imagename = "image_"+int2string(index)+".jpg";
        DrawRectDetectResultForImage(frame, detBoxs);   
        cv::imwrite(imagename, frame);
        index++;
      }

    DestoryDeepmodeInstance(pDNNInstance);	      
    std::cout << "Total detection time: " << total_time << "ms" <<"frame_num: "<<frame_num<< std::endl;
    std::cout << "Average fps: " << 1000 /(total_time / frame_num) << std::endl;     
    std::cout << "Finish !"<<std::endl;
    return 0;
}
