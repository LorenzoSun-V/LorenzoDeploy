/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-26 09:14:21
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-23 14:41:55
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <filesystem>
#include "utils.h"
#include "common.h"
#include "yoloseginfer.h"
using namespace std::filesystem;

bool isDirectoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

//新建目录
bool createDirectory(const std::string& path) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
        return true;
    } else {
        return false;
    }
}

bool writeSegmentResults(const std::vector<SegBox>& segResult, const std::string& filePath) {
    if (segResult.empty()) {
        // 如果 segResult 为空或只有一个 classID 为 -1 的元素，则返回 false
        return false;
    }
    if(segResult.size() == 1 && segResult[0].classID == -1){
        return false;
    }
    std::ofstream outfile(filePath);
    for(const auto& seg : segResult) {
        std::ostringstream oss;
        oss << seg.x << " " << seg.y << " " << seg.x + seg.w << " " << seg.y + seg.h << " " << seg.confidence << " " << seg.classID << "\n";
        std::string inputtext = oss.str();
        std::cout << inputtext;
        outfile << inputtext;
    }
    outfile.close();
    return true;
}

void writeMaskResults(const std::vector<cv::Mat>& masks, const std::string& maskFilePath) {
    if (masks.empty()) {
        std::cerr << "Error: No masks to write!" << std::endl;
        return;
    }
    std::ofstream outfile(maskFilePath, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening mask file!" << std::endl;
        return;
    }
    
    size_t numMasks = masks.size();
    // 写入 mask 的数量
    outfile.write(reinterpret_cast<const char*>(&numMasks), sizeof(numMasks));
    
    for (const auto& mask : masks) {
        int rows = mask.rows;
        int cols = mask.cols;
        // 写入每个 mask 的维度
        outfile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        outfile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        
        // 写入每个 mask 的数据
        outfile.write(reinterpret_cast<const char*>(mask.data), rows * cols * sizeof(float));
    }
    
    outfile.close();
}

std::string getBaseFileName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1); // 获取文件名部分
    }
    return path; // 如果路径中没有路径分隔符，则直接返回
}

int main(int argc, char* argv[])
{
    if(argc < 4) {
        // 0代表不使用yolov8，1代表使用yolov8
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

    std::string imageSavePath = std::string(pimagedir) + "_output";
    //检查图像保存文件夹不存在进行创建
    if (!isDirectoryExists(imageSavePath)) {
        if (createDirectory(imageSavePath)) {
            std::cout << "Directory created successfully!" << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory!" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Directory already exists!" << std::endl;
    }

    std::vector<std::string> imagePaths = getImagePaths(pimagedir);
    cv::Mat frame;
    std::vector<SegBox> segBoxs;
    std::vector<cv::Mat> masks;

    for (const std::string& imagePath : imagePaths) {
        std::cout << "图像路径: " << imagePath << std::endl;
        ReadFrameFromPath(imagePath.c_str(), frame);
        segBoxs.clear();
        masks.clear();
        std::cout << "推理中" << std::endl;
        InferenceGetInstanceSegmentResult(pDeepInstance, frame, segBoxs, masks);
        // if 
        // 将推理结果画在图像上保存下来
        std::cout << "绘制中" << std::endl;
        DrawInstanceSegmentResultForImage(frame, segBoxs, masks);  
        std::string imageOutPath = replaceImageOutPath(imagePath, "_out");  
        // 在文件名前面加上目录
        path p(imageOutPath);
        p.replace_filename(imageSavePath + p.filename().string());
        std::string newImageOutPath = p.string();
        std::cout << "图像存储地址: " << newImageOutPath << std::endl;
        cv::imwrite(newImageOutPath.c_str(), frame);

        // 将推理结果以文件形式存储到与图片同名的目录下
        std::string fileOutPath = replaceImageExtensionWithSuffix(imagePath);
        path p_seg(fileOutPath);
        p_seg.replace_filename(imageSavePath + p_seg.filename().string());
        std::string newfileOutPaht = p_seg.string();
        std::cout << "文件存储地址: " << newfileOutPaht << std::endl;
        bool save_result = writeSegmentResults(segBoxs, newfileOutPaht);

        // 只有当有检测结果时才保存mask
        if (save_result){
            std::string maskOutPath = replaceImageExtensionWithSuffix(imagePath, ".bin");
            path p_mask(maskOutPath);
            p_mask.replace_filename(imageSavePath + p_mask.filename().string());
            std::string newMaskOutPath = p_mask.string();
            std::cout << "mask存储地址: " << newMaskOutPath << std::endl;
            writeMaskResults(masks, newMaskOutPath);
            std::cout << "---------------------------------------" << std::endl;
        }
    }

    DestoryDeepmodeInstance(&pDeepInstance);	  

    return 0;
}