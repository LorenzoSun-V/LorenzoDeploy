/*
 * @FilePath: /bt_alg_api/test-Yaml/test/test.cpp
 * @Description: 接口代码
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-07 08:19:52
 */

#include <string>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include "yaml.h"

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
	    std::cout<<"example: ./binary conf.yaml "<<std::endl;
	    exit(-1);
    }

    const char* pYaml = argv[1];
    void* pYamlInstance = NULL; 
    //初始化获得句柄
    ENUM_ERROR_CODE eRet = InitYamlInstance(pYaml, &pYamlInstance);//9600 115200
    if(ENUM_OK != eRet || NULL == pYamlInstance) {
        std::cout<<"InitServerSocket "<<std::endl;
	    exit(-1);
    }
    //示例获得关键值    
    char context[64];
    if (GetYamlString(pYamlInstance, "InPutDevices", "devices", context, sizeof(context)) == 0) {
        printf("ClientName: %s\n", context);
    } else {
        printf("Key not found or error occurred while reading.\n");
    }
    memset(context, 0x00, sizeof(context));
    if (GetYamlString(pYamlInstance, "Server", "Port", context, sizeof(context)) == 0) {
        printf("Port: %s\n", context);
    } else {
        printf("Key not found or error occurred while reading.\n");
    }
    //配置服务器端口
    int new_port = 1777;
    printf("Set Port: %d\n", new_port);
    SetYamlString(pYamlInstance, "Server", "Port", std::to_string(new_port).c_str());
    //查看配置的端口
    if (GetYamlString(pYamlInstance, "Server", "Port", context, sizeof(context)) == 0) {
        printf("New Port: %s\n", context);
    } else {
        printf("Key not found or error occurred while reading.\n");
    }
    DestorytYamlInstance(&pYamlInstance);
    std::cout << "Finish !"<<std::endl;
    return 0;
}

// ./testyaml  ../../conf/conf.yaml