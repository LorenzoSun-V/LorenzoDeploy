/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-02-05 14:45:21
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-05-05 09:30:58
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
  YOLOv5Model model(yaml_path);
  // std::vector<std::string> batch_files = {
  //   "/lorenzo/deploy/LorenzoDeploy/fd/image/123/lQDPJwDerLc_-7HNASDNAYCwGt89uTCKll8GHPkepI_mAQ_384_288.jpg",
  //   "/lorenzo/deploy/LorenzoDeploy/fd/image/123/lQDPJx1BiY8BW7HNAeDNAoCwJOSsNR8rO9kGHPkepI_mAg_640_480.jpg",
  //   "/lorenzo/deploy/LorenzoDeploy/fd/image/123/lQDPJx1BiY8BW7HNAeDNAoCwJOSsNR8rO9kGHPkepI_mAg_940_480_2.jpg",
  //   "/lorenzo/deploy/LorenzoDeploy/fd/image/123/lQDPKGSikJBfjLHNASDNAYCwVtgNAqjAGwoGHPkepI_mAA_384_288.jpg",
  //   "/lorenzo/deploy/LorenzoDeploy/fd/image/123/lQDPKdbn6-Ixi7HNAoDNAoCwghvz9ZcIAp0GHPkepI_mAw_640_640.jpg",
  // };
  // model.InferImagesBatch(batch_files);
  model.InferFolder();
  return 0;
}