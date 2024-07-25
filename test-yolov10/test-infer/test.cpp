/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 16:20:55
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-03 15:22:38
 * @Description: YOLOv10 单batch测试代码
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
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
        std::cout<<"example: ./binary image_folder .bin"<<std::endl;
        exit(-1);
    }

    const char* pimagedir = argv[2];
    const char* pWeightsfile = argv[1];
        
    if(pimagedir == NULL || pWeightsfile == NULL){
        std::cout<<"input param error!"<<std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> frames; 
    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
    cv::Mat frame; 
    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        frames.push_back(frame);
    }

    void * pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 

    std::cout<<"Init Finshed!"<<std::endl;  
    int frame_num = static_cast<int>(frames.size());
    double total_time = 0.0;
    for (int i=0; i < frame_num; i++){
        std::vector<DetBox> detBoxs;
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pDNNInstance, frames[i], detBoxs);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
        total_time += t_detect_end - t_detect_start;
        std::string imagename = "image"+int2string(i)+".jpg";
        DrawRectDetectResultForImage(frames[i], detBoxs);   
        cv::imwrite(imagename, frames[i]);
    }

    std::cout << "Total detection time: " << total_time << "ms" << std::endl;
    std::cout << "Average fps: " << frame_num / total_time * 1000 << std::endl;
    std::cout << "Finish !"<<std::endl;
    DestoryDeepmodeInstance(pDNNInstance);
    return 0;
}