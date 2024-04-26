#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "fastdeploy/vision.h"
#include "common.h"

template<typename YOLOModel>
void InferImage(YOLOModel& model, const std::string& image_file, const std::string& output_folder);

template<typename YOLOModel>
void InferVideo(YOLOModel& model, const std::string& video_file, const std::string& output_folder);

void InferFolder(const Config& cfg);