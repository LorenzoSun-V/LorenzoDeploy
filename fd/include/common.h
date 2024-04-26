/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-04-25 15:04:45
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-04-26 16:16:16
 * @Description: 
 */
/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-04-25 15:04:45
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-04-26 08:45:13
 * @Description: 
 */
#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

struct Config {
    std::string model_path;    /*!< path to the model. */
    std::string source_folder; /*!< path to the source folder. */
    std::string output_folder; /*!< path to the output folder. */
    int run_option;            /*!< run option. */
    int img_size;              /*!< image size. */
    int bs;                    /*!< batch size. */
    float conf;               /*!< confidence threshold. */
    float nms_iou;            /*!< nms iou threshold. */
    float threshold;          /*!< threshold. */
};

inline void print_config(const Config& cfg) {
  std::cout << "==================== Config ====================" << std::endl;
  std::cout << "model_path: " << cfg.model_path << std::endl;
  std::cout << "source_folder: " << cfg.source_folder << std::endl;
  std::cout << "output_folder: " << cfg.output_folder << std::endl;
  std::cout << "run_option: " << cfg.run_option << std::endl;
  std::cout << "img_size: " << cfg.img_size << std::endl;
  std::cout << "bs: " << cfg.bs << std::endl;
  std::cout << "conf: " << cfg.conf << std::endl;
  std::cout << "nms_iou: " << cfg.nms_iou << std::endl;
  std::cout << "threshold: " << cfg.threshold << std::endl;
  std::cout << "===============================================" << std::endl;
}

inline Config load_config(const std::string& config_file){
  YAML::Node config = YAML::LoadFile(config_file);
  Config cfg;
  try {
    cfg.model_path = config["model_path"].as<std::string>();
    cfg.source_folder = config["source_folder"].as<std::string>();
    cfg.output_folder = config["output_folder"].as<std::string>();
    cfg.run_option = config["run_option"].as<int>();
    cfg.img_size = config["img_size"].as<int>();
    cfg.bs = config["bs"].as<int>();
    cfg.conf = config["conf"].as<float>();
    cfg.nms_iou = config["nms_iou"].as<float>();
    cfg.threshold = config["threshold"].as<float>();
  } 
  catch (YAML::Exception& e) {
    std::cerr << "Error parsing YAML: " << e.what() << std::endl;
    std::cerr << "Please check yaml config! Missing field!" << std::endl;
    throw; // 或者处理错误，比如返回一个错误码或默认配置
  }
  return cfg;
}