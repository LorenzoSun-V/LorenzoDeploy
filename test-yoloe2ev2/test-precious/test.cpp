/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 14:19:07
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-09-03 15:30:27
 * @Description: YOLOv10 精度处理代码
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
#include "utils.h"
#include "yoloe2ev2infer.h"
using namespace cv;
using namespace std;
using namespace std::filesystem;

//检查目录是否存在
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

void writeDetectionResults(const std::vector<DetBox>& detResult, const std::string& filePath) {
    std::ofstream outfile(filePath);
    for(const auto& det : detResult) {
        std::ostringstream oss;
        oss << det.x << " " << det.y << " " << det.x + det.w << " " << det.y + det.h << " " << det.confidence << " " << det.classID << "\n";
        std::string inputtext = oss.str();
        std::cout << inputtext;
        outfile << inputtext;
    }
    outfile.close();
}

int main(int argc, char* argv[]) 
{
    if(argc < 3){
        cout<<"param1: the make binary;"<<endl;
        cout<<"param2: input the detect image file;"<<endl;
        cout<<"param3: model path is the model and prototxt path;"<<endl;
        // cout<<"example: ./binary imagepath weightsfile "<<endl;
        exit(-1);
    }
    
    const char* folderPath = argv[1];
    const char* pWeightsfile = argv[2];
    
    cv::Mat frame;
    void * pDNNInstance= NULL; 
    ENUM_ERROR_CODE eOK =  LoadDeepModelModules(pWeightsfile, &pDNNInstance);
    if(eOK != ENUM_OK){
        cout<<"can not get pDNNInstance!"<<endl;
        return -1;
    } 
    std::string addition= "/output/";
    std::string imageSavePath = folderPath + addition;
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

    std::vector<std::string> imagePaths = getImagePaths(folderPath);
    cout<<"Init Finshed!"<<endl;  
    std::vector<DetBox> detResult;
    // 打印找到的图像路径
    for (const std::string& imagePath : imagePaths) {
        
        ReadFrameFromPath(imagePath.c_str(), frame);
        detResult.clear();
    
        InferenceGetDetectResult(pDNNInstance, frame, detResult);

        //将推理结果画在图像上保存下来
        DrawRectDetectResultForImage(frame, detResult);   
        std::string imageOutPath = replaceImageOutPath(imagePath, "_out");     
        // 在文件名前面加上目录
        path p(imageOutPath);
        p.replace_filename(imageSavePath + p.filename().string());
        std::string newImageOutPath = p.string();
        std::cout << "图像存储地址: " << newImageOutPath << std::endl;
        cv::imwrite(newImageOutPath.c_str(), frame);

        //将推理结果以文件形式存储到与图片同名的目录下
        std::string fileOutPath = replaceImageExtensionWithTxt(imagePath);
        path p2(fileOutPath);
        p2.replace_filename(imageSavePath + p2.filename().string());
        std::string newFileOutPath = p2.string();
        std::cout << "文件存储地址: " << newFileOutPath << std::endl;
        writeDetectionResults(detResult, newFileOutPath);
    }
    DestoryDeepmodeInstance(&pDNNInstance);           
    std::cout << "Finish !"<<endl;
    return 0;
}
