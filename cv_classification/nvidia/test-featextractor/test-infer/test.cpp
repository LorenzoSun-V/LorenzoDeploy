/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 10:20:03
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-09-03 17:37:58
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
#include "featextractor.h"

std::string int2string(int x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

std::string getFileName(const std::string path) {
    size_t lastSlash = path.find_last_of("/\\");
    size_t lastDot = path.find_last_of(".");
    if (lastDot == std::string::npos) {
        lastDot = path.length();
    }
    return path.substr(lastSlash + 1, lastDot - lastSlash - 1);
}

void saveFeaturesToFile(const std::vector<float>& features, const std::string filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        for (size_t i = 0; i < features.size(); ++i) {
            outFile << features[i];
            if (i != features.size() - 1) {
                outFile << " ";
            }
        }
        outFile << std::endl;
        outFile.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

int main(int argc, char* argv[])
{
    if(argc < 2) {
        std::cout<<"example: ./binary image_folder .bin"<<std::endl;
        exit(-1);
    }

    const char* pimagedir = argv[1];
    const char* pWeightsfile = argv[2];
        
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

    void* pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 

    std::cout<<"Init Finshed!"<<std::endl;  
    int frame_num = static_cast<int>(frames.size());
    double total_time = 0.0;
    for (int i=0; i < frame_num; i++){
        std::vector<float> objects;
        double t_detect_start = GetCurrentTimeStampMS();
        InferenceGetFeature(pDNNInstance, frames[i], objects);
        double t_detect_end = GetCurrentTimeStampMS();  
        fprintf(stdout, "inference time %.02lfms\n", t_detect_end - t_detect_start);
        total_time += t_detect_end - t_detect_start;

        // Save features to file with image name
        std::string imageFileName = getFileName(imagePaths[i]);
        std::string txtFileName = imageFileName + ".txt";
        saveFeaturesToFile(objects, txtFileName);
    }

    DestoryDeepmodeInstance(&pDNNInstance);
    std::cout << "Total detection time: " << total_time << "ms" << std::endl;
    std::cout << "Average fps: " << frame_num / total_time * 1000 << std::endl;
    std::cout << "Finish !"<<std::endl;
    return 0;
}