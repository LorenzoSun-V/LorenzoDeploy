/*
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/test-yolov8obb/test-batchinfer/test.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2025-01-24 10:23:33
 * @Description: YOLOv8OBB 多batch处理代码
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
        std::cout<<"example: ./binary image_folder .engine"<<std::endl;
        exit(-1);
    }

    const char* pWeightsfile = argv[1];
    const char* pimagedir = argv[2];

    if(pimagedir == NULL || pWeightsfile == NULL) {
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

    void * pDeepInstance= NULL; 
    ENUM_ERROR_CODE eOK = LoadDeepModelModules(pWeightsfile, &pDeepInstance);
    if(eOK != ENUM_OK) {
        std::cout<<"can not get pDeepInstance!"<<std::endl;
        return -1;
    } 
    std::cout<<"Init Finshed!"<<std::endl;  

    
    std::vector<std::vector<DetBox>> batchdetBoxs;
    double t_detect_start = GetCurrentTimeStampMS();
    BatchInferenceGetDetectResult(pDeepInstance, frames_list, batchdetBoxs);
    double t_detect_end = GetCurrentTimeStampMS();  

    fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start);

    int frame_num = static_cast<int>(frames_list.size());
    std::cout << "Total images: " << frame_num <<" detect result: " << batchdetBoxs.size() << std::endl;

    for (int i = 0; i < frame_num && batchdetBoxs.size() > 0; i++) {
        // 获取原始图像名称
        std::string baseImageName = getBaseFileName(imagePaths[i]);
   
        for (const auto& box : batchdetBoxs[i]) {
            std::cout<< baseImageName << ": x=" << box.x << ", y=" << box.y 
              << ", w=" << box.w << ", h=" << box.h 
              << ", angle=" << box.radian 
              << ", confidence=" << box.confidence 
              << std::endl;
        }

        // 生成新的图像名称，原始名称 + 下横线 + 序号
        std::string imagename = "_" + baseImageName.substr(0, baseImageName.find_last_of('.')) + ".jpg";
        DrawRotatedRectForImage(frames_list[i], batchdetBoxs[i]);   
        cv::imwrite(imagename, frames_list[i]);
    }
   
    std::cout << "Finish!" << std::endl;
    DestoryDeepmodeInstance(&pDeepInstance);
    return 0;
}
