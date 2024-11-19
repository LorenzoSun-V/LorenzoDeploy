/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 13:09:55
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-09-03 17:37:49
 * @Description: 特征提取 多batch测试代码
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
    if(eOK != ENUM_OK){
        std::cout<<"can not get pDNNInstance!"<<std::endl;
        return -1;
    } 
    
    std::vector<std::vector<float>> objects;
    std::cout<<"Init Finshed!"<<std::endl;  
 
    double t_detect_start = GetCurrentTimeStampMS();
    BatchInferenceGetFeature(pDNNInstance, batchframes, objects); 
    double t_detect_end = GetCurrentTimeStampMS();  
    int framenum = static_cast<int>(batchframes.size()); 

    std::cout << "Detected " << framenum << " frames." << std::endl;
    std::cout << "objects size: " << objects.size() << std::endl;
    for (int i=0; i < framenum; i++){
        std::string filename = getFileName(imagePaths[i]);
        std::cout << "Saving features to " << filename << ".txt" << std::endl;
        saveFeaturesToFile(objects[i], filename + ".txt");
    }

    DestoryDeepmodeInstance(&pDNNInstance);
    fprintf(stdout, "Total detection time %.02lfms\n", t_detect_end - t_detect_start);
    std::cout << "Average fps: " << framenum * 1000 / (t_detect_end - t_detect_start) << std::endl;	           
    std::cout << "Finish !"<<std::endl;
    return 0;
}

// export LD_LIBRARY_PATH=/home/bt/libs

// ./testbatch /home/mic-710aix/Downloads/valimage/val_imgs /home/mic-710aix/tensorrtx/yolov8/yolov8m_20240319_cls4_zs_v0.1.engine
