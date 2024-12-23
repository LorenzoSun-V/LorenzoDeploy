/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 10:20:03
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-20 15:48:05
 * @Description: 特征提取 单batch测试代码
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "utils.h"
#include "common.h"
#include "classificationInfer.h"

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
    void* pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 
    std::cout<<"Init Finshed!"<<std::endl;  


    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
    cv::Mat frame; 
    
    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        std::vector<ClsResult> cls_rets;
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetFeature(pDNNInstance, frame, cls_rets,1);
        double t_detect_end = GetCurrentTimeStampMS();  
      
        // 输出结果
        for (int i = 0; i < static_cast<int>(cls_rets.size()); ++i) {
            std::cout << "Class Score: " << cls_rets[i].score 
                    << ", Class ID: " << cls_rets[i].class_id << std::endl;
        }
        fprintf(stdout, "inference time %.02lfms\n", t_detect_end - t_detect_start);
    }

    DestoryDeepmodeInstance(&pDNNInstance);

    std::cout << "Finish !"<<std::endl;
    return 0;
}