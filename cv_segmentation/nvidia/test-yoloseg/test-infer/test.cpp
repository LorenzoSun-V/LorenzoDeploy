/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-26 09:14:21
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-17 15:54:37
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
#include "yoloseginfer.h"

std::string getBaseFileName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1); // 获取文件名部分
    }
    return path; // 如果路径中没有路径分隔符，则直接返回
}

int main(int argc, char* argv[])
{
    if(argc < 3) {
        std::cout<<"example: ./binary image_folder .bin 0"<<std::endl;
        exit(-1);
    }

    const char* pimagedir = argv[1];
    const char* pWeightsfile = argv[2];
    bool bUseYOLOv8 = std::string(argv[3]) == "1";
    
    if(pimagedir == NULL || pWeightsfile == NULL){
        std::cout<<"input param error!"<<std::endl;
        return -1;
    }
    
    void * pDeepInstance= NULL; 
    ENUM_ERROR_CODE eOK = LoadInstanceSegmentModelModules(pWeightsfile, &pDeepInstance, bUseYOLOv8);
    if(eOK != ENUM_OK){
        std::cout<<"can not get pDeepInstance!"<<std::endl;
        return -1;
    } 
    std::cout<<"Init Finshed!"<<std::endl;  

    std::vector<cv::Mat> batchframes; 
    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
    cv::Mat frame; 
    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        batchframes.push_back(frame);
    }
     
    int frame_num = static_cast<int>(batchframes.size());
    double total_time = 0.0;
    int index=1;
    std::vector<SegBox> segBoxs;
    std::vector<cv::Mat> masks;
    for( auto& frame: batchframes)
    {
        segBoxs.clear();
        masks.clear();
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetInstanceSegmentResult(pDeepInstance, frame, segBoxs, masks);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
        total_time += t_detect_end - t_detect_start;
        std::string imagename = "image_"+std::to_string(index)+".jpg";
        DrawInstanceSegmentResultForImage(frame, segBoxs, masks);  
        cv::imwrite(imagename, frame);
        index++;
      }

    DestoryDeepmodeInstance(&pDeepInstance);	  
    std::cout << "Total detection time: " << total_time << "ms" << std::endl;
    std::cout << "Average fps: " << frame_num / total_time * 1000 << std::endl;         
    std::cout << "Finish !"<<std::endl;
    return 0;
}