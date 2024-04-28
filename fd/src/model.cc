/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-04-26 14:21:37
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-04-28 17:06:19
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

namespace fs = std::filesystem;

// 基类Model的构造函数，加载配置文件并打印配置信息
Model::Model(const std::string& config_path){
    if (!fs::exists(config_path))
        std::cerr << "Config file not found: " << config_path << std::endl;
    cfg = load_config(config_path);
    print_config(cfg);
    if (cfg.threshold < 1 && cfg.threshold > 0)
        save_ori = true;
}

// 基类Model的configureRuntimeOptions函数，根据配置文件设置运行时选项
fastdeploy::RuntimeOption Model::configureRuntimeOptions() const {
    auto option = fastdeploy::RuntimeOption();
    if (cfg.run_option == 1 || cfg.run_option == 2) {
        option.UseGpu();
        if (cfg.run_option == 2) {
            option.UseTrtBackend();
            option.SetTrtInputShape("images", {cfg.bs, 3, cfg.img_size, cfg.img_size});
        }
    }
    return option;
}

/**
 * Processes an image file for object detection and saves the visualization and original image based on detection results and configuration settings.
 *
 * @param image_file The path to the image file to be processed.
 * 
 * This function reads an image from the specified file using OpenCV's imread function. It checks if the image
 * is loaded correctly. If not, it outputs an error message and returns early. If the image is successfully loaded,
 * it proceeds to predict objects in the image using the Predict function. If the prediction is successful,
 * it outputs the file name and prediction results to the console.
 *
 * After prediction, it visualizes the detection results on the image using the VisDetection function from FastDeploy,
 * saves this visualized image to the configured output folder with a filename prefixed by "vis_", and checks the configuration
 * to decide if the original image should be saved. If the `save_ori` flag is set and any detections exceed the
 * configured threshold, the original image is saved using the SaveOriginalImage function to the same output folder
 * with a filename prefixed by "ori_".
 */
void Model::InferImage(const std::string& image_file){
    auto image = cv::imread(image_file);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_file << std::endl;
        return;
    }
    fastdeploy::vision::DetectionResult res;
    if (!Predict(image, &res)) {
        std::cerr << "Failed to predict image: " << image_file << std::endl;
        return;
    }
    std::cout << "Image: " << image_file << std::endl;
    std::cout << res.Str() << std::endl;
    // 保存带有推理结果的图
    auto vis_image = fastdeploy::vision::VisDetection(image, res);
    fs::path output_path = fs::path(cfg.output_folder) / ("vis_" + fs::path(image_file).filename().string());
    cv::imwrite(output_path.string(), vis_image);

    // 保存满足阈值的原图
    if (save_ori){
        fs::path output_path_ori = fs::path(cfg.output_folder) / ("ori_" + fs::path(image_file).filename().string());
        SaveOriginalImage(image, output_path_ori, res);
    }
}

/**
 * Saves the original image to disk if any detected object in the image has a score above a defined threshold.
 * 
 * @param image The image to potentially save, provided as a constant reference to a cv::Mat object.
 * @param save_path The file path where the image should be saved if the condition is met.
 * @param res The detection results associated with the image, which includes scores, bounding boxes and classes.
 * 
 * This function iterates over the detection results. For each detected object, it checks if the score
 * of the detection exceeds a pre-set threshold defined in the configuration (`cfg.threshold`). If at least
 * one detection score is above the threshold, the function writes the image to the specified save_path
 * using OpenCV's imwrite function and then breaks out of the loop. This ensures the image is saved only once,
 * as soon as a qualifying detection is found.
 */
void Model::SaveOriginalImage(const cv::Mat& image, const std::string& save_path, const fastdeploy::vision::DetectionResult& res){
    for (size_t i = 0; i < res.boxes.size(); i++){
        if (res.scores[i] > cfg.threshold){
            cv::imwrite(save_path, image);
            break;
        }
    }
}

/**
 * Processes a batch of images for object detection, saves the visualizations and original images based on detection results and configuration settings, and clears the input and output containers.
 *
 * @param batch_images A reference to a vector containing the batch of images to be processed.
 * @param batch_names A reference to a vector containing the filenames of the images in the batch.
 * @param batch_results A pointer to a vector containing the detection results corresponding to the batch of images.
 * 
 * This function first calls the BatchPredict function to perform object detection on the batch of images.
 * If the prediction is unsuccessful, it outputs an error message and returns early.
 *
 * For each image in the batch, it visualizes the detection results using the VisDetection function from FastDeploy,
 * saves the visualized image to the configured output folder with a filename prefixed by "vis_", and checks the configuration
 * to decide if the original image should be saved. If the `save_ori` flag is set and any detections exceed the
 * configured threshold, the original image is saved using the SaveOriginalImage function to the same output folder
 * with a filename prefixed by "ori_".
 *
 * After processing all images in the batch, it clears the input and output containers to release memory.
 */
void Model::ProcessBatchImage(std::vector<cv::Mat>& batch_images, std::vector<std::string>& batch_names, std::vector<fastdeploy::vision::DetectionResult>* batch_results){
    if (!BatchPredict(batch_images, batch_results)) {
        std::cerr << "Failed to predict batch." << std::endl;
        return;
    }
    for (size_t i = 0; i < batch_images.size(); i++) {
        // 保存带有推理结果的图
        std::cout << batch_results->at(i).Str() << std::endl;
        auto vis_image = fastdeploy::vision::VisDetection(batch_images[i], batch_results->at(i));
        fs::path output_path = fs::path(cfg.output_folder) / ("vis_" + batch_names[i]);
        cv::imwrite(output_path.string(), vis_image);

        // 保存满足阈值的原图
        if (save_ori){
            fs::path output_path_ori = fs::path(cfg.output_folder) / ("ori_" + fs::path(batch_names[i]).filename().string());
            SaveOriginalImage(batch_images[i], output_path_ori.string(), batch_results->at(i));
        }
    }
    batch_images.clear();
    batch_names.clear();
    batch_results->clear();
}

/**
 * Processes a batch of image files for object detection, predicts objects, visualizes detection results, and saves the results to disk.
 * 
 * @param batch_files A vector containing the file paths of the images to be processed.
 * 
 * This function iterates through the provided vector of image file paths. For each image file, it attempts to load the image using OpenCV's imread function.
 * If loading fails, an error message is printed, and the function proceeds to the next image file.
 * 
 * For each successfully loaded image, it adds the image and its corresponding filename to separate vectors (batch_images and batch_names).
 * If the number of images in the batch reaches the specified batch size (cfg.bs), it calls the ProcessBatchImage function to perform object detection on the batch.
 * 
 * After processing all images in the batch_files vector, if there are any remaining images that did not form a complete batch, 
 * it calls ProcessBatchImage once more to process these remaining images.
 */
void Model::InferImagesBatch(const std::vector<std::string>& batch_files){
    std::vector<cv::Mat> batch_images;
    std::vector<std::string> batch_names;
    std::vector<fastdeploy::vision::DetectionResult> batch_results;
    batch_images.reserve(cfg.bs);
    batch_names.reserve(cfg.bs);

    for (const auto& file_path : batch_files) {
        auto image = cv::imread(file_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << file_path << std::endl;
            continue; // Skip this image and continue with the next
        }
        batch_images.push_back(image);
        batch_names.push_back(fs::path(file_path).filename().string());
        if (batch_images.size() == cfg.bs)
            ProcessBatchImage(batch_images, batch_names, &batch_results);
    }

    // 处理剩余的图片（如果有）
    if (!batch_images.empty())
        ProcessBatchImage(batch_images, batch_names, &batch_results);
}

/**
 * Processes a batch of video frames for object detection, predicts objects, visualizes detection results, and saves the results to disk.
 * 
 * @param batch_frames A reference to a vector containing the batch of video frames to be processed.
 * @param batch_results A pointer to a vector containing the detection results corresponding to the batch of video frames.
 * @param video_file The file path of the input video file.
 * @param video_writer A reference to the VideoWriter object used to write the visualized frames to an output video file.
 * @param num A pointer to an integer representing the frame number, used for naming the output images.
 * 
 * This function first calls the BatchPredict function to perform object detection on the batch of video frames.
 * If the prediction is unsuccessful, it outputs an error message and returns early.
 * 
 * For each frame in the batch, it visualizes the detection results using the VisDetection function from FastDeploy,
 * writes the visualized frame to the output video file using the provided VideoWriter object, and checks the configuration
 * to decide if the original frame and visualized frame should be saved as images. If the `save_ori` flag is set,
 * it saves the original frame and the visualized frame to the configured output folder with filenames 
 * based on the input video file name and the frame number.
 * 
 * After processing all frames in the batch, it clears the input and output containers to release memory.
 */
void Model::ProcessBatchVideo(
    std::vector<cv::Mat>& batch_frames, 
    std::vector<fastdeploy::vision::DetectionResult>* batch_results, 
    const std::string& video_file,
    cv::VideoWriter& video_writer, 
    int* num)
{
    if (!BatchPredict(batch_frames, batch_results)) {
        std::cerr << "Failed to predict batch." << std::endl;
        return;
    }
    for (size_t i = 0; i < batch_frames.size(); i++) {
        auto vis_frame = fastdeploy::vision::VisDetection(batch_frames[i], batch_results->at(i));
        video_writer.write(vis_frame);
        if (save_ori){
            fs::path output_path_ori = fs::path(cfg.output_folder) / (fs::path(video_file).stem().string() + "_" + std::to_string(*num) + ".jpg");
            fs::path output_path_vis = fs::path(cfg.output_folder) / (fs::path(video_file).stem().string() + "_" + std::to_string(*num) + "_vis.jpg");
            cv::imwrite(output_path_ori.string(), batch_frames[i]);
            cv::imwrite(output_path_vis.string(), vis_frame);
            (*num)++;
        }
    }
    batch_frames.clear();
    batch_results->clear();
}

/**
 * Processes a video file for object detection, visualizes detection results, and saves the results to an output video file.
 * 
 * @param video_file The file path of the input video file.
 * 
 * This function opens the input video file using OpenCV's VideoCapture class.
 * If the video file fails to open, it outputs an error message and returns early.
 * 
 * It retrieves the width, height, and frame rate of the input video using OpenCV's VideoCapture methods.
 * 
 * It constructs an output video file path in the specified output folder with a filename prefixed by "vis_".
 * 
 * It creates a VideoWriter object to write the visualized frames to the output video file using the specified codec (MJPG),
 * frame rate, and frame size.
 * 
 * It initializes vectors to store batch frames and detection results.
 * 
 * It enters a loop to read frames from the input video file until no more frames are available.
 * Within the loop:
 *  - It reads a frame from the input video file.
 *  - It checks if the frame is empty, indicating the end of the video.
 *  - It adds a deep copy of the frame to the batch_frames vector to ensure that the original frame is not modified during processing.
 *  - If the number of frames in the batch reaches the specified batch size (cfg.bs), it calls the ProcessBatchVideo function to perform object detection on the batch.
 * 
 * After processing all frames in the video, it releases the resources associated with the VideoCapture and VideoWriter objects.
 */
void Model::InferVideo(const std::string& video_file){
    cv::VideoCapture cap(video_file, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video file: " << video_file << std::endl;
        return;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    fs::path output_video_path = fs::path(cfg.output_folder) / ("vis_" + fs::path(video_file).filename().string());
    cv::VideoWriter video_writer(output_video_path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

    std::vector<cv::Mat> batch_frames;
    std::vector<fastdeploy::vision::DetectionResult> batch_results;
    batch_frames.reserve(cfg.bs);

    cv::Mat frame;
    int num = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            ProcessBatchVideo(batch_frames, &batch_results, video_file, video_writer, &num);
            break;
        }
        // 这里一定是深拷贝，因为frame会被修改
        batch_frames.push_back(frame.clone());
        if (batch_frames.size() == cfg.bs) {
            ProcessBatchVideo(batch_frames, &batch_results, video_file, video_writer, &num);
        }
    }

    cap.release();
    video_writer.release();
}

// YOLOv8Model的构造函数，调用基类Model的构造函数并初始化模型
void YOLOv8Model::InitModel(const Config& cfg) {
    auto option = configureRuntimeOptions();
    model = std::make_shared<fastdeploy::vision::detection::YOLOv8>(cfg.model_path, "", option);
    model->GetPreprocessor().SetSize({cfg.img_size, cfg.img_size});
    model->GetPostprocessor().SetConfThreshold(cfg.conf);
    model->GetPostprocessor().SetNMSThreshold(cfg.nms_iou);
    if (!model->Initialized())
        std::cerr << "Failed to initialize model." << std::endl;
}

bool YOLOv8Model::Predict(const cv::Mat& image, fastdeploy::vision::DetectionResult* res) {
    if (!model->Predict(image, res))
        return false;
    return true;
}

bool YOLOv8Model::BatchPredict(const std::vector<cv::Mat>& batch_images, std::vector<fastdeploy::vision::DetectionResult>* batch_results) {
    if (!model->BatchPredict(batch_images, batch_results))
        return false;
    return true;
}