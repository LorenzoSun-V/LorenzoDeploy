#include "postprocess.h"

static float iou(const InstanceSegResult& res1, const InstanceSegResult& res2) {
    float interBox[] = {
        std::max(res1.center_x - res1.w / 2.f, res2.center_x - res2.w / 2.f), // left
        std::min(res1.center_x + res1.w / 2.f, res2.center_x + res2.w / 2.f), // right
        std::max(res1.center_y - res1.h / 2.f, res2.center_y - res2.h / 2.f), // top
        std::min(res1.center_y + res1.h / 2.f, res2.center_y + res2.h / 2.f)  // bottom
    };

    // 检查是否没有重叠
    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    // 计算交集面积
    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);

    // 计算并集面积
    float unionBoxS = res1.w * res1.h + res2.w * res2.h - interBoxS;

    // 返回 IoU
    return interBoxS / unionBoxS;
}

static bool cmp(const InstanceSegResult& a, const InstanceSegResult& b) {
    if (a.score == b.score) {
        return a.center_x < b.center_x;
    }
    return a.score > b.score;
}

cv::Mat scale_mask(cv::Mat mask, cv::Mat img, int model_height, int model_width) {
    if (mask.empty() || img.empty()) {
        std::cerr << "mask or img is empty." << std::endl;
    }
    int x, y, w, h;
    float r_w = model_width / (img.cols * 1.0);
    float r_h = model_height / (img.rows * 1.0);
    if (r_h > r_w) {
        w = model_width;
        h = r_w * img.rows;
        x = 0;
        y = (model_height - h) / 2;
    } else {
        w = r_h * img.cols;
        h = model_height;
        x = (model_width - w) / 2;
        y = 0;
    }
    cv::Rect r(x, y, w, h);
    cv::Mat res;
    cv::resize(mask(r), res, img.size());
    return res;
}

bool batch_nms(std::vector<std::vector<InstanceSegResult>>& batch_res, float* output, model_param_t model_param, bool m_buseyolov8)
{

    if(nullptr == output){
        std::cerr << "output is NULL." << std::endl;
        return false;
    }

    batch_res.resize(model_param.batch_size);

    // 遍历每个batch
    for (int batch_idx = 0; batch_idx < model_param.batch_size; ++batch_idx) 
    {
        std::map<int, std::vector<InstanceSegResult>> class_bboxes;
        // 遍历当前batch的所有检测框
        for (int i = 0; i < model_param.num_bboxes; ++i) {
            float confidence = -1.0f;
            int class_id = -1;

            if (m_buseyolov8) {
                // 找到当前检测框中最高类别置信度
                for (int c = 0; c < model_param.num_classes; ++c) {
                    float class_conf = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 4 + c]; // YOLOv8类别置信度从第5个字段开始
                    if (class_conf > confidence) {  // YOLOv8置信度就是模型输出的置信度
                        confidence = class_conf;
                        class_id = c;
                    }
                }
                // 忽略低置信度框
                if (confidence < model_param.conf_thresh) continue;

                // 解码检测框信息
                InstanceSegResult box;
                box.center_x = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 0];  
                box.center_y = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 1];  
                box.w = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 2]; 
                box.h = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 3];  
                for (int j = 0; j < 32; j++) {
                    box.mask[j] = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 4 + model_param.num_classes + j];
                }
                box.score = confidence;
                box.class_id = class_id;  

                // 将检测框保存到对应类别的集合
                class_bboxes[class_id].push_back(box);
            }
            else {
                float obj_conf = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 4]; // YOLOv5目标置信度在第5个字段
                // 找到当前检测框中最高类别置信度
                for (int c = 0; c < model_param.num_classes; ++c) {
                    float class_conf = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 5 + c]; // YOLOv5类别置信度从第6个字段开始
                    if (obj_conf * class_conf > confidence) {  // YOLOv5最终的置信度是目标置信度和类别置信度的乘积
                        confidence = obj_conf * class_conf;
                        class_id = c;
                    }
                }

                // 忽略低置信度框
                if (confidence < model_param.conf_thresh) continue;

                // 解码检测框信息
                InstanceSegResult box;
                box.center_x = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 0];  
                box.center_y = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 1];  
                box.w = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 2]; 
                box.h = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 3];  
                for (int j = 0; j < 32; j++) {
                    box.mask[j] = output[(batch_idx * model_param.num_bboxes + i) * model_param.bbox_element + 5 + model_param.num_classes + j];
                }
                box.score = confidence;
                box.class_id = class_id;  

                // 将检测框保存到对应类别的集合
                class_bboxes[class_id].push_back(box);
            }
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
                    if (iou(bboxes[i], bboxes[j]) >= model_param.iou_thresh) {
                        suppressed[j] = true; // 抑制 IoU 大于阈值的框
                    }
                }
            }
        }

    }
    return true;
}

bool postprocess_batch(
    const std::vector<std::vector<InstanceSegResult>>& batch_bboxes,
    const std::vector<std::vector<cv::Mat>>& batch_masks,
    const std::vector<cv::Mat>& batch_images, 
    int model_width, 
    int model_height, 
    std::vector<std::vector<SegBox>>& batch_bboxes_result,
    std::vector<std::vector<cv::Mat>>& batch_masks_result) 
{

    if(batch_images.empty() ) {
        std::cerr << "images is NULL." << std::endl;
        return false;
    }
    if(batch_masks.empty()) {
        std::cerr << "masks is NULL." << std::endl;
        return false;
    }
    if(batch_bboxes.empty()) {
        std::cerr << "bboxes result is NULL." << std::endl;
        return false;
    }

    // clear and pre-allocate space to avoid repeated dynamic expansion
    batch_bboxes_result.clear();
    batch_masks_result.clear();
    batch_bboxes_result.reserve(batch_bboxes.size());
    batch_masks_result.reserve(batch_bboxes.size());

    // traverse each image and its corresponding bboxes
    for (size_t batch_idx = 0; batch_idx < batch_bboxes.size() && batch_idx < batch_images.size(); ++batch_idx) {

        // avoid copying by using const reference
        const auto& bboxes = batch_bboxes[batch_idx];
        const auto& image  = batch_images[batch_idx];
        const auto& masks  = batch_masks[batch_idx];

        // 2.1) process and scale masks first
        std::vector<cv::Mat> scaled_masks;
        scaled_masks.reserve(masks.size()); // pre-allocate space
        for (const auto& mask : masks) {
            scaled_masks.push_back(scale_mask(mask, image, model_height, model_width));
        }
        // save all masks of the current batch to batch_masks_result
        batch_masks_result.push_back(std::move(scaled_masks));

        // 2.2) if no detection box in the current batch, fill in empty data and skip
        if (bboxes.empty()) {
            batch_bboxes_result.push_back({ SegBox() });
            continue;
        }

        // calculate scale factor and padding，计算缩放因子和填充
        float r_w = static_cast<float>(model_width)  / image.cols;
        float r_h = static_cast<float>(model_height) / image.rows;
        float image_y_pad = (model_height - r_w * image.rows) * 0.5f;
        float image_x_pad = (model_width  - r_h * image.cols) * 0.5f;

        // 2.3) traverse all detection boxes in the current batch and convert coordinates， 遍历当前批次的所有检测框并转换坐标
        std::vector<SegBox> detBoxForBatch;
        detBoxForBatch.reserve(bboxes.size()); // pre-allocate space，预先分配空间
        for (const auto& box_in : bboxes)
        {
            // top-left / bottom-right， 坐标转换为左上角和右下角坐标
            float origin_x1 = box_in.center_x - box_in.w * 0.5f;
            float origin_y1 = box_in.center_y - box_in.h * 0.5f;
            float origin_x2 = box_in.center_x + box_in.w * 0.5f;
            float origin_y2 = box_in.center_y + box_in.h * 0.5f;

            // 把坐标转换为相对于原图的坐标
            float x1, y1, x2, y2;
            if (r_h > r_w) {
                x1 = origin_x1;
                x2 = origin_x2;
                y1 = origin_y1 - image_y_pad;
                y2 = origin_y2 - image_y_pad;
                x1 /= r_w;  x2 /= r_w;
                y1 /= r_w;  y2 /= r_w;
            } else {
                x1 = origin_x1 - image_x_pad;
                x2 = origin_x2 - image_x_pad;
                y1 = origin_y1;
                y2 = origin_y2;
                x1 /= r_h;  x2 /= r_h;
                y1 /= r_h;  y2 /= r_h;
            }

            // save the converted box to detBoxForBatch，保存转换后的框到detBoxForBatch
            SegBox box_out;
            box_out.x         = x1;          // top-left x
            box_out.y         = y1;          // top-left y
            box_out.w         = (x2 - x1);   
            box_out.h         = (y2 - y1);
            box_out.confidence = box_in.score;
            box_out.classID   = box_in.class_id;

            detBoxForBatch.push_back(std::move(box_out));
        }

        batch_bboxes_result.push_back(std::move(detBoxForBatch));
    }

    return true;
}

static cv::Rect get_downscale_rect(const InstanceSegResult bbox, float scale) {
    float left = bbox.center_x - bbox.w / 2.0f;
    float top = bbox.center_y - bbox.h / 2.0f;
    float right = bbox.center_x + bbox.w / 2.0f;
    float bottom = bbox.center_y + bbox.h / 2.0f;

    left /= scale;
    top /= scale;
    right /= scale;
    bottom /= scale;

    return cv::Rect(
        round(left),                   // top-left x
        round(top),                    // top-left y
        round(right - left),           // width
        round(bottom - top)            // height
    );
}

bool batch_process_mask(
    const float* proto, 
    int proto_size, 
    const std::vector<std::vector<InstanceSegResult>>& batch_dets, 
    std::vector<std::vector<cv::Mat>>& batch_masks, 
    const model_param_t& model_param) 
{
    if (nullptr == proto) {
        std::cerr << "proto is NULL." << std::endl;
        return false;
    }
    batch_masks.clear();

    // offset of proto for each batch
    size_t batch_seg_output_size = model_param.seg_output * model_param.seg_output_height * model_param.seg_output_width;

    // traverse each batch
    for (size_t batch_idx = 0; batch_idx < batch_dets.size(); batch_idx++) {
        // offset of proto for the current batch
        const float* proto_offset = proto + batch_idx * batch_seg_output_size;
        // used to save masks for the current batch
        std::vector<cv::Mat> masks_for_batch;
        // traverse each detection box in the current batch
        for (const auto& det : batch_dets[batch_idx]) {
            cv::Mat mask_mat = cv::Mat::zeros(model_param.seg_output_height, model_param.seg_output_width, CV_32FC1);
            auto r = get_downscale_rect(det, 4);
            for (int x = r.x; x < r.x + r.width; x++) {
                for (int y = r.y; y < r.y + r.height; y++) {
                    float e = 0.0f;
                    for (int j = 0; j < 32; j++) {
                        e += det.mask[j] * proto_offset[j * proto_size / 32 + y * mask_mat.cols + x];
                    }
                    e = 1.0f / (1.0f + expf(-e));
                    mask_mat.at<float>(y, x) = e;
                }
            }
            cv::resize(mask_mat, mask_mat, cv::Size(model_param.input_width, model_param.input_height));
            masks_for_batch.push_back(mask_mat);
        }
        batch_masks.push_back(std::move(masks_for_batch));
    }
    return true;
}
