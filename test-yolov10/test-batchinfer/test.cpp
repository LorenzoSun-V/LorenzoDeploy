/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 13:09:55
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-03 09:02:42
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include "utils.h"
#include "common.h"
#include "yolov10infer.h"

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
    if(argc < 2) {
          std::cout<<"example: ./binary imagedir .engine"<<std::endl;
          exit(-1);
    }

    const char* pimagedir = argv[1];
    const char* pWeightsfile = argv[2];
          
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
    
    std::vector<std::vector<DetBox>> batchDetBoxs;
    std::cout<<"Init Finshed!"<<std::endl;  
 
    double t_detect_start = GetCurrentTimeStampMS();
    BatchInferenceGetDetectResult(pDNNInstance, batchframes, batchDetBoxs); 
    double t_detect_end = GetCurrentTimeStampMS();  
    size_t ndetboxs = batchDetBoxs.size();
    int framenum = static_cast<int>(batchframes.size()); 
    
    for(int i=0; i < framenum && ndetboxs >0; i++){
        std::string imagename = "image_"+int2string(i)+".jpg";
        DrawRectDetectResultForImage(batchframes[i], batchDetBoxs[i]);   
        cv::imwrite(imagename, batchframes[i]);
    }

    DestoryDeepmodeInstance(pDNNInstance);
    fprintf(stdout, "Total detection time %.02lfms\n", t_detect_end - t_detect_start);
    std::cout << "Average fps: " << framenum * 1000 / (t_detect_end - t_detect_start) << std::endl;	           
    std::cout << "Finish !"<<std::endl;
    return 0;
}

// export LD_LIBRARY_PATH=/home/bt/libs

// ./testbatch /home/mic-710aix/Downloads/valimage/val_imgs /home/mic-710aix/tensorrtx/yolov8/yolov8m_20240319_cls4_zs_v0.1.engine