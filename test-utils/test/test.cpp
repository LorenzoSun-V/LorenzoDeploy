
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "utils.h"
#include "common.h"

int main(int argc, char* argv[])
{
    cv::Rect detbox;
    detbox.x=138;
    detbox.y=0;
    detbox.width=61;
    detbox.height=287;
    
    std::vector<cv::Point> converted_points;

    converted_points.push_back({28,0});
    converted_points.push_back({319,0});
    converted_points.push_back({28,253});
    converted_points.push_back({319,253});
    bool bret = CalculateAreaRatio(detbox, converted_points, 50);
    std::cout <<"bret= "<<bret<<std::endl;
    return 0;
}

