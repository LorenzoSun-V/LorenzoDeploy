/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-02-05 14:45:21
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-04-29 09:29:09
 * @Description: 
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision.h"
#include "model.h"


int main(int argc, char* argv[]) {
  if (argc!=2){
    std::cerr << "Usage: ./detect path/to/yaml" << std::endl;
    return -1;
  }
  std::string yaml_path = argv[1];
  Config cfg = load_config(yaml_path);
  YOLOv8Model model(yaml_path);
  model.InferFolder();
  return 0;
}