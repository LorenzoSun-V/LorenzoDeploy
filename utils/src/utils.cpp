#include "utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <vector>

//获取当前时刻毫秒级时间
double GetCurrentTimeStampMS()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

//读取图片接口
bool ReadFrameFromPath(const char* pImagePath, cv::Mat& frame)
{
    if( pImagePath == NULL )
    {
        std::cout << "please check input image path!" <<  std::endl;
	    return false;
    }
    
    frame = cv::imread(pImagePath);
    if( frame.empty() )
    {
        std::cout<<"image is empty!"<< std::endl;
        return false;
    }
    return true;
}

// 弧度转换为角度
float radianToDegree(float radian) {
    return radian * 180.0f / static_cast<float>(M_PI);
}

// 绘制旋转矩形框
void DrawRotatedRectForImage(cv::Mat &image, const std::vector<DetBox> detBoxs) 
{
    if (image.empty()) {
        std::cout << "Frame is empty." << std::endl;
        return;
    }

    if (detBoxs.empty()) {
        std::cout << "detBoxs is empty!" << std::endl;
        return;
    }

    // 生成随机颜色
    std::vector<cv::Scalar> color;
    color.push_back(cv::Scalar(0, 0, 255));
    color.push_back(cv::Scalar(0, 255, 0));
    color.push_back(cv::Scalar(255, 0, 0));
    color.push_back(cv::Scalar(0, 255, 255));

    // 绘制每个检测框
    for (const auto& obj : detBoxs) {
        // 创建旋转矩形对象
        cv::Point2f center(obj.x+obj.w/2, obj.y+obj.h/2);
        cv::Size2f size(obj.w, obj.h);
        float angle = radianToDegree(obj.radian);
        cv::RotatedRect rotatedRect(center, size, angle);

        // 获取旋转矩形的四个顶点
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);

        // 绘制旋转矩形
        for (int j = 0; j < 4; j++) {
            cv::line(image, vertices[j], vertices[(j + 1) % 4], color[static_cast<int>(obj.classID) % color.size()], 2);
        }

        // 在框的附近显示类别ID和置信度
        std::string label = std::to_string(obj.classID) + ": " + std::to_string(obj.confidence);
        cv::putText(image, label, cv::Point(obj.x, obj.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

//在图像上绘制目标框
ENUM_ERROR_CODE DrawRectDetectResultForImage(cv::Mat &frame, std::vector<DetBox> detBoxs)
{
    if (frame.empty())
    {
        std::cout <<  "frame is empty." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    
    if( detBoxs.empty() )
    {
    	std::cout << "detBoxs is empty!" << std::endl;
	    return ERR_INVALID_PARAM;
    }
     
	//生成随机颜色
    std::vector<cv::Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
	}
    //将ID和置信度显示上去
    for (size_t i = 0; i < detBoxs.size(); i++)
    {
        const DetBox& obj = detBoxs[i];
        int labelID = obj.classID;
        cv::rectangle(frame, cv::Rect(obj.x, obj.y, obj.w, obj.h), color[obj.classID], 2, 8);
        std::string strid = std::to_string(labelID) +':'+ std::to_string(obj.confidence);
        cv::putText(frame, strid, cv::Point(obj.x+10, obj.y+25), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1, 8, 0);               
    }

    return ENUM_OK;
}  


//ip地址正确性检查
int CheckValidIPAddress(const std::string& ipaddr) 
{
    std::regex ip_regex("(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})");
    std::smatch ip_match;

    if (std::regex_search(ipaddr, ip_match, ip_regex)) {
        for (int i = 1; i <= 4; ++i) {
            int octet = std::stoi(ip_match[i]);
            if ( !(octet >= 0 && octet <= 255) ) {
                 std::cout << "Check camera ip address is invalid, part is: " + std::to_string(octet) << std::endl;
                 return -1;
            }
        }
    } else {
        std::cout <<"IP address not found" << std::endl;
        return -2;
    }
    return 0;
}

// IP地址检查，提取IP地址并存储到外部的unsigned char数组中
int ExtractIPAddress(const std::string& input, unsigned char ip[4]) 
{
    std::regex ip_regex("(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})");
    std::smatch ip_match;

    if (std::regex_search(input, ip_match, ip_regex)) {
        for (int i = 1; i <= 4; ++i) {
            int octet = std::stoi(ip_match[i]);
            if (octet >= 0 && octet <= 255) {
                ip[i - 1] = static_cast<unsigned char>(octet);
            } else {
                 std::cout << "Invalid IP address part: " + std::to_string(octet) << std::endl;
                 return -1;
            }
        }
    } else {
        std::cout <<"IP address not found" << std::endl;
        return -2;
    }
    return 0;
}


//字符串转16进制    	    	  
void str2Hex(char *pDest, char *pSrc, int nSize)
{
    char h1,h2;
	char s1,s2;
	int i;

	for (i=0; i<nSize; i++)
	{
        h1 = pSrc[2*i];
        h2 = pSrc[2*i+1];

        s1 = h1 - '0';
        if (s1 > 9) 
        s1 -= 7;

        s2 = h2 - '0';
        if (s2 > 9) 
        s2 -= 7;

        pDest[i] = (s1<<4) | s2;
	}  
}

void ConvertStringToHexArray(char *pDest, const char *pSrc, int nSize) 
{
    char h1, h2;
    int i;

    for (i = 0; i < nSize; i++) {
        h1 = pSrc[2*i];
        h2 = pSrc[2*i + 1];

        // Convert ASCII to nibble
        if (h1 >= '0' && h1 <= '9') h1 -= '0';
        else if (h1 >= 'A' && h1 <= 'F') h1 -= 'A' - 10;
        else if (h1 >= 'a' && h1 <= 'f') h1 -= 'a' - 10;
        else return; // Invalid character

        if (h2 >= '0' && h2 <= '9') h2 -= '0';
        else if (h2 >= 'A' && h2 <= 'F') h2 -= 'A' - 10;
        else if (h2 >= 'a' && h2 <= 'f') h2 -= 'a' - 10;
        else return; // Invalid character

        pDest[i] = (h1 << 4) | h2;
    }
}


//综合逻辑判断，计算目标框与检测区域的百分比，大于阈值返回true 
bool CalculateAreaRatio(const cv::Rect detbox, const std::vector<cv::Point> converted_points, int threshold)
{    
    if(converted_points.empty() ) {
        std::cout <<"CalculateAreaRatio "<<std::endl;
        return false;
    }

    // 矩形转为多边形
    std::vector<cv::Point> box_point;
    box_point.push_back(detbox.tl());
    box_point.push_back(cv::Point(detbox.br().x, detbox.tl().y));
    box_point.push_back(cv::Point(detbox.tl().x, detbox.br().y));
    box_point.push_back(detbox.br());
    if(box_point.empty() )
    {
        std::cout <<"box_point "<<std::endl;
        return false;
    }

    // 计算交集坐标点
    std::vector<cv::Point> intersect_point;
    cv::intersectConvexConvex(box_point, converted_points, intersect_point);
    if(intersect_point.empty() )
    {
        std::cout <<"intersect_point "<<std::endl;
        return false;
    }

    // 计算交集面积
    double intersectArea = cv::contourArea(intersect_point);

    // 计算矩形面积
    double boxArea = detbox.area();

    // 计算并返回百分比
    int boxValue = (intersectArea / boxArea) * 100;
    if(boxValue > threshold) return true;
    return false;
}

//输入模型检测的图像和检测区域的坐标点，检测区域的长宽，将检测区域的坐标转为与输入原始图像相同尺寸的坐标
bool NormalizedPointsToImageSize( int image_width, int image_height, std::vector<cv::Point> area_points, int area_width, int area_height, std::vector<cv::Point> &converted_points)
{
    if(image_width < 320 || image_height < 240){
        printf("Input image width(%d) height(%d) must more than 320,240", image_width, image_height);
        return false;
    }    

    if(area_width < 320 || area_height < 240){
        printf("Input area point width(%d) height(%d) must more than 320,240", area_width, area_height);
        return false;
    }    

    //配置原始图像缩放因子
    float scale_w = (float)area_width / image_width;
    float scale_h = (float)area_height / image_height;

    for(auto point: area_points)
    {
        int x = point.x/ scale_w;
        int y = point.y / scale_h;
        converted_points.push_back({x,y});
    }

    return true;
}

bool endsWithImage(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin(), 
        [](char a, char b) { return tolower(a) == tolower(b); });
}

// 遍历输入文件夹的所有图片
std::vector<std::string> getImagePaths(const std::string& folder) {
    std::vector<std::string> imagePaths;
    DIR* dir = opendir(folder.c_str());
    if (dir == nullptr) {
        std::cerr << "Could not open directory: " << folder << std::endl;
        return imagePaths;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string path = folder + "/" + entry->d_name;
        struct stat s;
        if (stat(path.c_str(), &s) == 0 && S_ISREG(s.st_mode)) { 
            // Regular file
            if (endsWithImage(path, ".jpg") || endsWithImage(path, ".png") || 
                endsWithImage(path, ".jpeg") || endsWithImage(path, ".bmp")) {
                imagePaths.push_back(path);
            }
        }
    }
    closedir(dir);
    return imagePaths;
} 

//将图片名称后缀替换对应的文件后缀
std::string replaceImageExtensionWithTxt(const std::string& path) {
    std::vector<std::string> extensions = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"};
    std::string newPath = path;

    for (const std::string& ext : extensions) {
        size_t pos = newPath.rfind(ext);
        if (pos != std::string::npos && pos == newPath.length() - ext.length()) {
            newPath.replace(pos, ext.length(), ".txt");
            break;
        }
    }

    return newPath;
}

//将图片名称追加结果输出图像字段
std::string replaceImageOutPath(const std::string& path, std::string suffix_name) {
    std::vector<std::string> extensions = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"};
    std::string newPath = path;

    for (const std::string& ext : extensions) {
        size_t pos = newPath.rfind(ext);
        if (pos != std::string::npos && pos == newPath.length() - ext.length()) {
            newPath.replace(pos, ext.length(), suffix_name + ext);
            break;
        }
    }

    return newPath;
}

// 判断一个点是否在多边形内
bool isPointInPolygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon)
{
    return pointPolygonTest(polygon, point, false) >= 0;
}

// 判断一个点是否在旋转矩形内
bool isPointInRotatedRect(const cv::Point2f& point, const cv::RotatedRect& rect) 
{
    cv::Point2f center = rect.center;

    float angle = rect.angle * CV_PI / 180.0;

    float cosA = cos(-angle);
    float sinA = sin(-angle);

    float dx = point.x - center.x; 
    float dy = point.y - center.y; 

    float localX = dx * cosA - dy * sinA;
    float localY = dx * sinA + dy * cosA;

    float halfWidth = rect.size.width / 2.0;
    float halfHeight = rect.size.height / 2.0;

    return (abs(localX) <= halfWidth) && (abs(localY) <= halfHeight);
}

// 获取旋转矩形在多边形内所有点和交集部分的面积
float getPointsInRotatedRectorArea(Rotates& Rotating) 
{
    cv::Rect boundingBox = Rotating.rotatedRect.boundingRect();

    for (int y = boundingBox.y; y < boundingBox.y + boundingBox.height; y++) {
        for (int x = boundingBox.x; x < boundingBox.x + boundingBox.width; x++) {
            if (x >= 0 && x < Rotating.width && y >= 0 && y < Rotating.height) 
            {
                if (isPointInRotatedRect(cv::Point2f(x, y), Rotating.rotatedRect)) 
                {
                    Rotating.points.emplace_back(x, y);
                }
            }
        }
    }

    float intersectionArea = 0.0f;
    for (const auto& pt : Rotating.points) {
        if (isPointInPolygon(cv::Point2f(pt.x, pt.y), Rotating.polygon)) {
            intersectionArea += 1.0f; // 计算交集区域的点数
            Rotating.pointsInPolygon.push_back(pt); // 把点坐标顺便记下来.
        }
    }
    float rectArea = Rotating.size.width * Rotating.size.height;
    float intersectionPercentage = (intersectionArea / rectArea) * 100.0f;

    return intersectionPercentage;
}

// 绘制多边形和旋转矩形
void drawAndSaveTemperatureMap(const Rotates& Rotating, const std::string& filename) 
{
    // 创建一个临时Mat
    cv::Mat colorMap(288, 384, CV_8UC3, cv::Scalar(128, 128, 128));

    for (const auto& pt : Rotating.points) {
        colorMap.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(0, 255, 255); // 黄色
    }

    for (const auto& pt : Rotating.pointsInPolygon) {
        colorMap.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255, 0, 0); // 蓝色
    }

    for (size_t i = 0; i < Rotating.polygon.size(); i++) {
        line(colorMap, Rotating.polygon[i], Rotating.polygon[(i + 1) % Rotating.polygon.size()], cv::Scalar(0, 255, 0), 2); // 绿色
    }

    imwrite(filename, colorMap);
    std::cout << "Saved temperature map to " << filename << std::endl;
}

// 获取旋转框内的最大温度
float getMaxTemperature(const std::vector<cv::Point>& points, const float temp[384][288]) 
{
    float maxTemp = -FLT_MAX;

    for (const auto& pt : points) {
        float tempValue = temp[pt.x][pt.y];
        maxTemp = std::max(maxTemp, tempValue);
    }
    return maxTemp;
}