/**
* @file      test.cpp
*
* @brief     精度验证推理代码
*
* @copyright 无锡宝通智能科技股份有限公司
*
* @author  图像算法组-贾俊杰
*
* All Rights Reserved.
*/
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include "rkinfer.h"
#include "utils.h"

using namespace cv;
using namespace std;

#define SAVE_OUTPUT_FILE
#define SAVE_OUTPUT_IMAGE
//#define SHOW_DETECT_IMAGE


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

int main(int argc, char* argv[]) 
{
	if(argc < 3){
		cout<<"param1: the make binary;"<<endl;
		cout<<"param2: model path is the model and prototxt path;"<<endl;
		cout<<"param3: input the detect image file;"<<endl;
		cout<<"example: ./binary weightsfile imagepath"<<endl;
		exit(-1);
 	}
    
  const char* pWeightsfile = argv[1];
  const char* folderPath = argv[2];
  //模型推理获得结果
  void * pRkInferInstance = NULL;
  ENUM_ERROR_CODE eRet = InitRKInferenceInstance(pWeightsfile,2, &pRkInferInstance);
  if(pRkInferInstance == NULL || eRet != ENUM_OK){
      cout<<"can not get pRkInferInstance!"<<endl;
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

  std::vector<DetBox> detResult;
  std::vector<std::string> imagePaths = getImagePaths(folderPath);
  cv::Mat frame;  
  cout<<"Init Finshed!"<<endl;  
  for (const std::string& imagePath : imagePaths) {
      std::cout << "图像路径: " << imagePath << std::endl;
      
      ReadFrameFromPath(imagePath.c_str(), frame);
      detResult.clear();
      double t_detect_start = GetCurrentTimeStampMS();
      InferenceGetDetectResult(pRkInferInstance, frame, detResult);
      double t_detect_end = GetCurrentTimeStampMS();  

      //将结果画到图上再保存下来
      fprintf(stdout, "detection time %.02lfms\n", t_detect_end - t_detect_start); 
      //将推理结果画在图像上保存下来
      DrawRectDetectResultForImage(frame, detResult);  	
#ifdef SAVE_OUTPUT_FILE 
      std::string imageOutPath = replaceImageOutPath(imagePath, "_output");   
      // 在文件名前面加上目录
      size_t imagefound = imageOutPath.find_last_of("/");
      std::string newImageOutPath;
      if (imagefound != std::string::npos) {
          //保存路径添加输出目录
          newImageOutPath = imageOutPath.substr(0, imagefound + 1) + addition + imageOutPath.substr(imagefound + 1);
          std::cout << "图像存储地址: " << newImageOutPath << std::endl;
      } else {
          std::cerr << "Invalid file path!" << std::endl;
          return 1;
      }
      cv::imwrite(newImageOutPath.c_str(), frame);
#endif
#ifdef SAVE_OUTPUT_IMAGE 
      //将推理结果以文件形式存储到与图片同名的目录下
      std::string fileOutPath = replaceImageExtensionWithTxt(imagePath);
      size_t filefound = fileOutPath.find_last_of("/");
      std::string newFileOutPath;
      if (filefound != std::string::npos) {
          //保存路径添加输出目录
          newFileOutPath = fileOutPath.substr(0, filefound + 1) + addition + fileOutPath.substr(filefound + 1);
          std::cout << "文件存储地址: " << newFileOutPath << std::endl;
      } else {
          std::cerr << "Invalid file path!" << std::endl;
          return 1;
      }
      std::ofstream outfile(newFileOutPath);
      int detsize = static_cast<int>(detResult.size());
      for(int i =0; i < detsize; i++)
      {
          std::string inputtext= std::to_string(detResult[i].x) + " " + std::to_string(detResult[i].y) + " " \
          + std::to_string(detResult[i].x+detResult[i].w) + " "+std::to_string(detResult[i].y+detResult[i].h) + " " \
          + std::to_string(detResult[i].confidence) + " " + std::to_string(detResult[i].classID) + "\n";
          std::cout << inputtext << std::endl;
          outfile<< inputtext;
      }
      outfile.close();
#endif
#ifdef SHOW_DETECT_IMAGE
  cv::imshow("frame", frame);
  cv::waitKey(5);
#endif
  }
   
  std::cout << "Finish !"<<endl;
  return 0;
}




