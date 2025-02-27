/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2025-01-07 08:40:33
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-24 10:06:13
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

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int main(int argc, char* argv[])
{
    if(argc < 2) {
        std::cout<<"example: ./binary image_folder .engine"<<std::endl;
        exit(-1);
    }

    const char* pimagedir = argv[2];
    const char* pWeightsfile = argv[1];
        
    if(pimagedir == NULL || pWeightsfile == NULL) {
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

    void * pDeepInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDeepInstance);
    if(eOK != ENUM_OK) {
        std::cout<<"can not get pDeepInstance!"<<std::endl;
        return -1;
    } 

    std::cout<<"Init Finshed!"<<std::endl;  

    int frame_num = static_cast<int>(frames.size());
    double total_time = 0.0;
    for (int i=0; i < frame_num; i++){
        std::vector<DetBox> detBoxs;
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pDeepInstance, frames[i], detBoxs);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
        total_time += t_detect_end - t_detect_start;
        // 生成新的图像名称，原始名称 + 下横线 + 序号
        std::string baseImageName = getBaseFileName(imagePaths[i]);
        std::string imagename = "_" + baseImageName.substr(0, baseImageName.find_last_of('.')) + ".jpg";
        DrawRectDetectResultForImage(frames[i], detBoxs);   
        cv::imwrite(imagename, frames[i]);
    }

    // std::cout << "Total detection time: " << total_time << "ms" << std::endl;
    // std::cout << "Average fps: " << frame_num / total_time * 1000 << std::endl;
    std::cout << "Finish !"<<std::endl;
    DestoryDeepmodeInstance(&pDeepInstance);
    return 0;
}
