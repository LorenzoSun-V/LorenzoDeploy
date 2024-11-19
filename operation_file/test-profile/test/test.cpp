/*
 * @FilePath: /bt_alg_api/test-profile/test/test.cpp
 * @Description: 接口代码
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-06 14:02:06
 */
/**
* @file    test.cpp
*
* @brief    配置文件读写测试代码
*
* @copyright 无锡宝通智能科技股份有限公司
*
* @author  图像算法组-贾俊杰
*
* All Rights Reserved.
*/
#include <string>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include "profile.h"

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
	    std::cout<<"example: ./binary conf.ini "<<std::endl;
	    exit(-1);
    }

    const char* pProfile = argv[1];
    void* pProfileInstance = NULL; 
    //初始化获得句柄
    ENUM_ERROR_CODE eRet = InitProfileInstance(pProfile, &pProfileInstance);//9600 115200
    if(ENUM_OK != eRet || NULL == pProfileInstance) {
        std::cout<<"InitServerSocket "<<std::endl;
	    exit(-1);
    }
    //示例获得关键值    
    char context[64];
    if (GetProfileString(pProfileInstance, "InPutDevices", "devices", context, sizeof(context)) == 0) {
        printf("ClientName: %s\n", context);
    } else {
        printf("Key not found or error occurred while reading.\n");
    }
    memset(context, 0x00, sizeof(context));
    if (GetProfileString(pProfileInstance, "Server", "Port", context, sizeof(context)) == 0) {
        printf("Port: %s\n", context);
    } else {
        printf("Key not found or error occurred while reading.\n");
    }
    //配置服务器端口
    SetProfileString(pProfileInstance, "Server", "Port", "8989");
    //查看配置的端口
    if (GetProfileString(pProfileInstance, "Server", "Port", context, sizeof(context)) == 0) {
        printf("New Port: %s\n", context);
    } else {
        printf("Key not found or error occurred while reading.\n");
    }
    DestorytProfileInstance(&pProfileInstance);
    std::cout << "Finish !"<<std::endl;
    return 0;
}

// ./testprofile  ../../conf/conf.ini