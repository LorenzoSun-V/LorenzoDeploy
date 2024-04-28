/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-02-05 14:45:21
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-04-28 16:20:45
 * @Description: 
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision.h"
#include "common.h"
#include "model.h"
#include "detect.h"


int main(int argc, char* argv[]) {
  if (argc!=2){
    std::cerr << "Usage: ./detect path/to/yaml" << std::endl;
    return -1;
  }
  std::string yaml_path = argv[1];
  Config cfg = load_config(yaml_path);
  // InferFolder(cfg);
  YOLOv8Model model(yaml_path);
  std::string a = "/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_16.jpg";
  model.InferImage(a);
  std::vector<std::string> b = {"/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_7.jpg",
                                "/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_16.jpg",
                                "/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_41.jpg",
                                "/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_43.jpg",
                                "/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_49.jpg",
                                "/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_56.jpg",
                                "/data/bt/kjg_multi/raw/new/nanlutian/20240424/images/1714022029855_57.jpg",};
  model.InferImagesBatch(b);
  std::string video_path = "/data/bt/kjg_multi/raw_zips/nanlutian/20240424/slide/20240424102119461_6.mp4";
  model.InferVideo(video_path);
  return 0;
}