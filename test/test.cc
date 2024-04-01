/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-03-29 16:42:31
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-03-29 17:07:01
 * @Description: 
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    const char* image_path = argv[1];
    // 读取图片并转换为灰度图
    Mat pic_color = imread(image_path);
    if(pic_color.empty()) {
        cout << "Image not found!" << endl;
        return -1;
    }
    Mat pic_gray;
    cvtColor(pic_color, pic_gray, COLOR_BGR2GRAY);

    // 显示灰度图的大小
    cout << "Size of gray image: " << pic_gray.rows << " x " << pic_gray.cols << endl;

    // 计算相邻列之间的相关性
    vector<double> co(pic_gray.cols, 1.0);
    for(int cols = 1; cols < pic_gray.cols; ++cols) {
        Mat A = pic_gray.col(cols-1);
        Mat B = pic_gray.col(cols);
        Scalar A_mean = mean(A);
        Scalar B_mean = mean(B);
        cout<<A_mean[0]<<endl;
        A = A - A_mean[0];
        B = B - B_mean[0];

        double numerator = A.dot(B);
        double denominator = sqrt(sum(A.mul(A))[0] * sum(B.mul(B))[0]);
        co[cols] = (denominator != 0) ? numerator / denominator : 0;
    }

    // 应用最小值滤波
    int filt_width = 20;
    vector<double> co_min_filt(pic_gray.cols, 1.0);
    for(int cols = filt_width; cols < pic_gray.cols; ++cols) {
        auto start = co.begin() + cols - filt_width + 1;
        auto end = co.begin() + cols + 1;
        co_min_filt[cols] = *min_element(start, end);
    }

    // 应用最小值滤波并保存结果
    ofstream outFile1("co.txt");
    if (!outFile1.is_open()) {
        cout << "Failed to open file for writing." << endl;
        return -1;
    }

    for(const auto& val : co) {
        outFile1 << val << "\n";
    }

    outFile1.close();

    // 应用最小值滤波并保存结果
    ofstream outFile("co_min_filt.txt");
    if (!outFile.is_open()) {
        cout << "Failed to open file for writing." << endl;
        return -1;
    }

    for(const auto& val : co_min_filt) {
        outFile << val << "\n";
    }

    outFile.close();

    // 在C++中，显示图像比较简单，但绘制曲线图比较复杂，需要额外的图形库支持
    // 以下仅展示如何显示原图
    // imwrite("res.png", pic_gray);
    // 对于曲线图，你可能需要将数据导出到文件，然后使用Python或其他工具绘图
    // 或者使用专门的C++图形库，如Qt或matplotlib-cpp

    return 0;
}
