/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-26 09:14:21
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-24 10:25:24
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
    for(int i=0; i<frame_num; i++)
    {
        frame = batchframes[i];
        segBoxs.clear();
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetInstanceSegmentResult(pDeepInstance, frame, segBoxs, masks);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
        total_time += t_detect_end - t_detect_start;
        // 生成新的图像名称，原始名称 + 下横线 + 序号
        std::string baseImageName = getBaseFileName(imagePaths[i]);
        std::string imagename = "_" + baseImageName.substr(0, baseImageName.find_last_of('.')) + ".jpg";
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