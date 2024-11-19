/*
 * @FilePath: /bt_zs_31/include/cv_utils.h
 * @Description: 接口代码
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-01-12 14:33:17
 */
#pragma once

#include <opencv2/opencv.hpp>
#include "common.h"
#include <deque>
//图像缓存队列
class ImageCacheQueue {
public:
    ImageCacheQueue(size_t maxSize) : _maxSize(maxSize) {}
    //视频帧添加进缓存队列中
    void addFrame(const cv::Mat frame) {
        if (_queue_mat_frame.size() >= _maxSize) {
            _queue_mat_frame.pop_front(); // 如果队列满了，移除最早的帧
        }
        _queue_mat_frame.push_back(frame.clone()); // 添加新帧到队列
    }
    
    std::vector<cv::Mat> getCacheFrames() {
        // 将队列中的所有帧一次性复制到 vector 中
        return std::vector<cv::Mat>(_queue_mat_frame.begin(), _queue_mat_frame.end());
    }

    size_t size() const {
        return _queue_mat_frame.size();
    }

private:
    std::deque<cv::Mat> _queue_mat_frame;
    size_t _maxSize;
};

