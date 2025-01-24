/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 13:09:55
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-24 10:13:55
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
    if(eOK != ENUM_OK){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 
    
    std::vector<std::vector<DetBox>> batchDetBoxs;
    std::cout<<"Init Finshed!"<<std::endl;  
 
    double t_detect_start = GetCurrentTimeStampMS();
    BatchInferenceGetDetectResult(pDNNInstance, batchframes, batchDetBoxs); 
    double t_detect_end = GetCurrentTimeStampMS();  
    size_t ndetboxs = batchDetBoxs.size();
    int frame_num = static_cast<int>(batchframes.size()); 
    
    for(int i=0; i < frame_num && ndetboxs >0; i++){
        // 生成新的图像名称，原始名称 + 下横线 + 序号
        std::string baseImageName = getBaseFileName(imagePaths[i]);
        std::string imagename = "_" + baseImageName.substr(0, baseImageName.find_last_of('.')) + ".jpg";
        DrawRectDetectResultForImage(batchframes[i], batchDetBoxs[i]);   
        cv::imwrite(imagename, batchframes[i]);
    }
    
    double total_time =  t_detect_end - t_detect_start;
    DestoryDeepmodeInstance(&pDNNInstance);
    fprintf(stdout, "Total detection time %.02lfms\n", total_time);
    std::cout << "Average fps: " << 1000 /(total_time / frame_num) << std::endl; 

    std::cout << "Finish !"<<std::endl;
    return 0;
}
