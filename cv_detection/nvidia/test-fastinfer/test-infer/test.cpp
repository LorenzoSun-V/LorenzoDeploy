/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 14:19:07
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-10-12 09:07:30
 * @Description: YOLOv10 精度处理代码，增加图像旋转、保存文件名及 tif 支持，tif 转为 jpg
 */
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "fastinfer.h"

using namespace cv;
using namespace std;
using namespace std::filesystem;

bool endsWithImage(const std::string& str, const std::vector<std::string>& suffixes) {
    for (const std::string& suffix : suffixes) {
        if (str.size() >= suffix.size() && 
            std::equal(suffix.rbegin(), suffix.rend(), str.rbegin(), 
            [](char a, char b) { return tolower(a) == tolower(b); })) {
            return true;
        }
    }
    return false;
}

// 遍历输入文件夹的所有图片，包括tif格式
std::vector<std::string> getImagePaths(const std::string& folder) {
    std::vector<std::string> imagePaths;
    std::vector<std::string> supportedFormats = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"};
    DIR* dir = opendir(folder.c_str());
    if (dir == nullptr) {
        std::cerr << "Could not open directory: " << folder << std::endl;
        return imagePaths;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string path = folder + "/" + entry->d_name;
        struct stat s;
        if (stat(path.c_str(), &s) == 0 && S_ISREG(s.st_mode)) { 
            // Regular file
            if (endsWithImage(path, supportedFormats)) {
                imagePaths.push_back(path);
            }
        }
    }
    closedir(dir);
    return imagePaths;
}

// 将文件路径中的扩展名更改为 .jpg
std::string changeExtensionToJpg(const std::string& filePath) {
    return filesystem::path(filePath).replace_extension(".jpg").string();
}

int main(int argc, char* argv[]) 
{
    if(argc < 4){
        cout<<"param1: the make binary;"<<endl;
        cout<<"param2: input the detect image folder;"<<endl;
        cout<<"param3: model path for weights and prototxt;"<<endl;
        cout<<"param4: output folder for detected images;"<<endl;
        exit(-1);
    }
    
    const char* folderPath = argv[1];
    const char* pWeightsfile = argv[2];
    const char* pNewfolder = argv[3]; 

    // 检查并创建 pNewfolder 目录
    if (!filesystem::exists(pNewfolder)) {
        if (!filesystem::create_directories(pNewfolder)) {
            std::cerr << "Failed to create directory: " << pNewfolder << std::endl;
            return -1;
        }
    }

    cv::Mat frame;
    void* pDNNInstance = LoadDeepModelModules(pWeightsfile, NULL, 0, MODEL_CODE_9, RUN_TRT);
    if(!pDNNInstance) {
        cout<<"Cannot get pDNNInstance!"<<endl;
        return -1;
    } 

    std::vector<std::string> imagePaths = getImagePaths(folderPath);
    cout<<"Initialization Finished!"<<endl;  

    DetBox *pDetBox = new DetBox[100];
    if (pDetBox == NULL) {
        std::cout<<"pDetBox is NULL!"<<std::endl;
        return -1;
    }
    int detCount = 0;

    // 处理每张图片
    for (const std::string& imagePath : imagePaths) {
        
        ReadFrameFromPath(imagePath.c_str(), frame);
        
        // 旋转图像90度
      //  cv::rotate(frame, frame, ROTATE_90_CLOCKWISE);
        
        InferenceGetDetectResult(pDNNInstance, frame, pDetBox, &detCount);

        // 将推理结果画在图像上保存下来
        RectDetectResultForImage(frame, pDetBox, detCount); 
        
        // 保存推理有结果的原始图像到指定文件夹，转为 jpg 格式
        if(detCount > 0){
            string outputFilePath = string(pNewfolder) + "/" + changeExtensionToJpg(path(imagePath).filename().string());  // 转换为 jpg 扩展名
            cv::imwrite(outputFilePath, frame);
        }
    }
    
    DestoryDeepmodeInstance(&pDNNInstance);           
    std::cout << "Processing Finished!" << std::endl;
    return 0;
}
