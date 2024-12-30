/*
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/yolov8obb/utils/postprocess.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-13 09:48:36
 */
#include "postprocess.h"

std::tuple<float, float, float> convariance_matrix(BBox res) {
    float w = res.w;
    float h = res.h;

    float a = w * w / 12.0;
    float b = h * h / 12.0;
    float c = res.radian;

    float cos_r = std::cos(c);
    float sin_r = std::sin(c);

    float cos_r2 = cos_r * cos_r;
    float sin_r2 = sin_r * sin_r;

    float a_val = a * cos_r2 + b * sin_r2;
    float b_val = a * sin_r2 + b * cos_r2;
    float c_val = (a - b) * cos_r * sin_r;

    return std::make_tuple(a_val, b_val, c_val);
}

static float probiou(const BBox& res1, const BBox& res2, float eps = 1e-7) {
    // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    float a1, b1, c1, a2, b2, c2;
//  std::tuple<float, float, float> matrix1 = {a1, b1, c1};
//     std::tuple<float, float, float> matrix2 = {a2, b2, c2};
    std::tie(a1, b1, c1) = convariance_matrix(res1);
    std::tie(a2, b2, c2) = convariance_matrix(res2);

    float x1 = res1.center_x, y1 = res1.center_y;
    float x2 = res2.center_x, y2 = res2.center_y;

    float t1 = ((a1 + a2) * std::pow(y1 - y2, 2) + (b1 + b2) * std::pow(x1 - x2, 2)) /
               ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps);
    float t3 = std::log(
            ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2)) /
                    (4 * std::sqrt(std::max(a1 * b1 - c1 * c1, 0.0f)) * std::sqrt(std::max(a2 * b2 - c2 * c2, 0.0f)) +
                     eps) +
            eps);

    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = std::max(std::min(bd, 100.0f), eps);
    float hd = std::sqrt(1.0 - std::exp(-bd) + eps);

    return 1 - hd;
}

static bool cmp(const BBox& a, const BBox& b) {
    if (a.score == b.score) {
        return a.center_x < b.center_x;
    }
    return a.score > b.score;
}

void nms_obb(std::vector<BBox>& res, float* output,  model_param_t model_param) {
    if(NULL == output) {
        std::cerr << "output is NULL." << std::endl;
        return ;
    }
    //int bbox_element = 4 + num_classes + 1; // cx, cy, w, h + 2*class + radian
    std::map<int, std::vector<BBox>> class_bboxes;

    // 遍历所有检测框
    for (int i = 0; i < model_param.num_bboxes; ++i) {
        float confidence = -1.0f;
        int class_id = -1;

        // 找到当前检测框中最高类别置信度
        for (int c = 0; c < model_param.num_classes; ++c) {
            float class_conf = output[i * model_param.bbox_element + 4 + c]; // 类别置信度从第5个字段开始
            if (class_conf > confidence) {
                confidence = class_conf;
                class_id = c;
            }
        }

        // 忽略低置信度框
        if (confidence < model_param.conf_thresh) continue;

        // 解码检测框信息
        BBox box;
        box.center_x = output[i * model_param.bbox_element + 0];  
        box.center_y = output[i * model_param.bbox_element + 1];  
        box.w = output[i * model_param.bbox_element + 2]; 
        box.h = output[i * model_param.bbox_element + 3];  
        box.radian = output[i * model_param.bbox_element + 4 + model_param.num_classes];  
        box.score = confidence; // 将softmax以后比较获得最大置信度保存下来
        box.class_id = class_id;  

        // 将检测框保存到对应类别的集合
        class_bboxes[class_id].push_back(box);
    }

    // 对每个类别执行 NMS
    for (auto& [class_id, bboxes] : class_bboxes) {
        // 按置信度降序排序
        std::sort(bboxes.begin(), bboxes.end(), cmp);

        std::vector<bool> suppressed(bboxes.size(), false);
        
        for (size_t i = 0; i < bboxes.size(); ++i) {
            if (suppressed[i]) continue;

            res.push_back(bboxes[i]);  

            for (size_t j = i + 1; j < bboxes.size(); ++j) {
                if (probiou(bboxes[i], bboxes[j]) >= model_param.iou_thresh) {
                    suppressed[j] = true; // 抑制 IoU 大于阈值的框

                }
            }
        }
    }
}

bool postprocess(std::vector<BBox> bboxes, cv::Mat image, int model_width, int model_height, std::vector<DetBox>& result) 
{
    if(bboxes.empty()) {
        std::cerr << "bboxes result is NULL." << std::endl;
        return false;
    }

    if(image.empty()) {
        std::cerr << "image is empty." << std::endl;
        return false;
    }

    float r_w = model_width/(image.cols * 1.0);
    float r_h = model_height/(image.rows * 1.0);
    float image_y_pad = (model_height - r_w * image.rows) / 2;
    float image_x_pad = (model_width - r_h * image.cols) / 2;

    float x1, x2, y1, y2;
    for (int i = 0; i < bboxes.size(); i++)
    {
        float origin_x1 =  bboxes[i].center_x - bboxes[i].w/2;
        float origin_y1 =  bboxes[i].center_y - bboxes[i].h/2;
        float origin_x2 =  bboxes[i].center_x + bboxes[i].w/2;
        float origin_y2 =  bboxes[i].center_y + bboxes[i].h/2;

        if (r_h > r_w) {
            x1 = origin_x1; 
            x2 = origin_x2;
            y1 = origin_y1 - image_y_pad;
            y2 = origin_y2 - image_y_pad;
            x1 = x1 / r_w;
            x2 = x2 / r_w;
            y1 = y1 / r_w;
            y2 = y2 / r_w;
        } else {
            x1 = origin_x1  - image_x_pad; 
            x2 = origin_x2  - image_x_pad;
            y1 = origin_y1;
            y2 = origin_y2;
            x1 = x1 / r_h;
            x2 = x2 / r_h;
            y1 = y1 / r_h;
            y2 = y2 / r_h;
        }

        DetBox box;
        box.x = x1;
        box.y = y1;
        box.w = x2 - x1;
        box.h = y2 - y1; 
        box.confidence = bboxes[i].score;
        box.classID = bboxes[i].class_id;
        box.radian =  bboxes[i].radian;
        // Add the DetBox to the detBoxs vector
        result.emplace_back(box);
    }

    if(result.empty()) return false;

    return true;
}


void nms_obb_batch(std::vector<std::vector<BBox>>& batch_res, float* output,  model_param_t model_param)
{
    if(NULL == output){
        std::cerr << "output is NULL." << std::endl;
        return ;
    }
    
    //float conf_thresh, float nms_thresh, int num_classes, int num_bboxes, int bbox_element, int batch_size) {
    // batch_res 用于存储每个批次的NMS结果
    batch_res.resize(model_param.batch_size);

    // 遍历每个batch
    for (int batch_idx = 0; batch_idx < model_param.batch_size; ++batch_idx) 
    {

        std::map<int, std::vector<BBox>> class_bboxes;
        // 遍历当前batch的所有检测框
        for (int i = 0; i < model_param.num_bboxes; ++i) {
            float confidence = -1.0f;
            int class_id = -1;

            // 找到当前检测框中最高类别置信度
            for (int c = 0; c < model_param.num_classes; ++c) {
                float class_conf = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 4 + c]; // 类别置信度从第5个字段开始
                if (class_conf > confidence) {
                    confidence = class_conf;
                    class_id = c;
                }
            }

            // 忽略低置信度框
            if (confidence < model_param.conf_thresh) continue;

            // 解码检测框信息
            BBox box;
            box.center_x = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 0];  
            box.center_y = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 1];  
            box.w = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 2]; 
            box.h = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 3];  
            box.radian = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 4 + model_param.num_classes];  
            box.score = confidence; // 将softmax以后比较获得最大置信度保存下来
            box.class_id = class_id;  

            // 将检测框保存到对应类别的集合
            class_bboxes[class_id].push_back(box);
        }

        // 对每个类别执行 NMS
        for (auto& [class_id, bboxes] : class_bboxes) {
            // 按置信度降序排序
            std::sort(bboxes.begin(), bboxes.end(), cmp);

            std::vector<bool> suppressed(bboxes.size(), false);

            for (size_t i = 0; i < bboxes.size(); ++i) {
                if (suppressed[i]) continue;

                batch_res[batch_idx].push_back(bboxes[i]);

                for (size_t j = i + 1; j < bboxes.size(); ++j) {
                    if (probiou(bboxes[i], bboxes[j]) >= model_param.iou_thresh) {
                        suppressed[j] = true; // 抑制 IoU 大于阈值的框
                    }
                }
            }
        }

    }
}

bool postprocess_batch(std::vector<std::vector<BBox>> batch_bboxes, 
                 std::vector<cv::Mat> batch_images, 
                 int model_width, 
                 int model_height, 
                 std::vector<std::vector<DetBox>>& batch_result) 
{

    if(batch_images.empty() ) {
        std::cerr << "bboxes result is NULL." << std::endl;
        return false;
    }

    if(batch_bboxes.empty()) {
        std::cerr << "bboxes result is NULL." << std::endl;
        return false;
    }

    // 遍历每个图像及其对应的 bboxes
    for (int batch_idx = 0; batch_idx < batch_bboxes.size(); batch_idx++) {
        const std::vector<BBox>& bboxes = batch_bboxes[batch_idx];
        const cv::Mat& image = batch_images[batch_idx];

        // 检查每个图像的 bboxes 是否为空
        if (bboxes.empty()) {
            //std::cerr << "result data for batch " << batch_idx << " is NULL." << std::endl;
            DetBox emptyBox;
            batch_result.push_back({ emptyBox });
            continue;  // 跳过当前批次的后续处理，继续下一个批次
        }

        float r_w = model_width / (image.cols * 1.0);
        float r_h = model_height / (image.rows * 1.0);
        float image_y_pad = (model_height - r_w * image.rows) / 2;
        float image_x_pad = (model_width - r_h * image.cols) / 2;

        float x1, x2, y1, y2;
        std::vector<DetBox> detBoxForBatch;

        // 遍历当前批次的所有检测框
        for (int i = 0; i < bboxes.size(); i++) {
            float origin_x1 = bboxes[i].center_x - bboxes[i].w / 2;
            float origin_y1 = bboxes[i].center_y - bboxes[i].h / 2;
            float origin_x2 = bboxes[i].center_x + bboxes[i].w / 2;
            float origin_y2 = bboxes[i].center_y + bboxes[i].h / 2;

            // 计算坐标，适应图片的缩放和填充
            if (r_h > r_w) {
                x1 = origin_x1;
                x2 = origin_x2;
                y1 = origin_y1 - image_y_pad;
                y2 = origin_y2 - image_y_pad;
                x1 = x1 / r_w;
                x2 = x2 / r_w;
                y1 = y1 / r_w;
                y2 = y2 / r_w;
            } else {
                x1 = origin_x1 - image_x_pad;
                x2 = origin_x2 - image_x_pad;
                y1 = origin_y1;
                y2 = origin_y2;
                x1 = x1 / r_h;
                x2 = x2 / r_h;
                y1 = y1 / r_h;
                y2 = y2 / r_h;
            }

            // 创建 DetBox 并填充数据
            DetBox box;
            box.x = x1;
            box.y = y1;
            box.w = x2 - x1;
            box.h = y2 - y1;
            box.confidence = bboxes[i].score;
            box.classID = bboxes[i].class_id;
            box.radian = bboxes[i].radian;

            // 将当前 batch 的 DetBox 添加到 detBoxForBatch
            detBoxForBatch.emplace_back(box);
        }

        // 将当前批次的 detBox 添加到 batch_res 中
        batch_result.emplace_back(detBoxForBatch);
    }


    return true;
}

