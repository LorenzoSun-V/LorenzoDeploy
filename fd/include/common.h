/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-04-25 15:04:45
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-04-26 13:43:54
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

// Config load_config(const std::string& config_file);

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