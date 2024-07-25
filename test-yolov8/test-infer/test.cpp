/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 13:09:55
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-17 10:33:23
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>

#include <sys/stat.h>
#include "utils.h"
#include "common.h"
#include "yolov8infer.h"

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
    if(argc < 2) {
          std::cout<<"example: ./binary .bin imagedir"<<std::endl;
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
    std::cout << "Total detection time: " << total_time << "ms" << std::endl;
    std::cout << "Average fps: " << frame_num / total_time * 1000 << std::endl;         
    std::cout << "Finish !"<<std::endl;
    return 0;
}

