/*
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/test-yolov8obb/test-infer/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-12 16:33:07
 * @Description: YOLOv8OBB 单batch处理代码
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "utils.h"
#include "common.h"
#include "yolov8obbinfer.h"

std::string getBaseFileName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1); // 获取文件名部分
    }
    return path; // 如果路径中没有路径分隔符，则直接返回
}

int main(int argc, char* argv[])
{
    if(argc < 2) {
        std::cout<<"example: ./binary image_folder .bin"<<std::endl;
        exit(-1);
    }

    const char* pWeightsfile = argv[1];
    const char* pimagedir = argv[2];

    if(pimagedir == NULL || pWeightsfile == NULL){
        std::cout<<"input param error!"<<std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> frames_list; 
    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
    cv::Mat frame; 
    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        frames_list.push_back(frame);
    }

    void * pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK = LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 

    std::cout<<"Init Finshed!"<<std::endl;  
    int frame_num = static_cast<int>(frames_list.size());

    double total_time = 0.0;
    for (int i=0; i < frame_num; i++){
        std::vector<DetBox> detBoxs;
        detBoxs.clear();
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetDetectResult(pDNNInstance, frames_list[i], detBoxs);
        double t_detect_end = GetCurrentTimeStampMS();  
        //fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);
        
        // 获取原始图像名称
        std::string baseImageName = getBaseFileName(imagePaths[i]);

        for (const auto& box : detBoxs) {
        std::cout << baseImageName << " : x=" << box.x << ", y=" << box.y 
              << ", w=" << box.w << ", h=" << box.h 
              << ", angle=" << box.radian 
              << ", confidence=" << box.confidence 
              << std::endl;
        }

        total_time += t_detect_end - t_detect_start;


        // 生成新的图像名称，原始名称 + 下横线 + 序号
        std::string imagename = "_" + baseImageName.substr(0, baseImageName.find_last_of('.')) + ".jpg";

        DrawRotatedRectForImage(frames_list[i], detBoxs); 
        cv::imwrite(imagename, frames_list[i]);
    }

    std::cout << "Infer Finish !"<<std::endl;
    DestoryDeepmodeInstance(&pDNNInstance);
    return 0;
}
