#pragma once

#include <string>

struct Config {
    std::string model_path;    /*!< path to the model. */
    std::string source_folder; /*!< path to the source folder. */
    std::string output_folder; /*!< path to the output folder. */
    int run_option;            /*!< run option. */
    int img_size;              /*!< image size. */
    float conf;               /*!< confidence threshold. */
    float nms_iou;            /*!< nms iou threshold. */
    float threshold;          /*!< threshold. */
};