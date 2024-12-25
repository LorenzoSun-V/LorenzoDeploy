/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-11-06 13:23:50
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-11-07 11:09:17
 * @Description: yaml文件读取接口
 */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "yaml.h"

class YamlParam {
public:
    bool bParamIsOk;
    const char *yaml_file;
    YAML::Node config;

    YamlParam() {
        bParamIsOk = false;
    }
};

class YamlInstance {
public:
   std::shared_ptr<YamlParam> _param;

   YamlInstance() {
        _param = std::make_shared<YamlParam>();
    }
};

/**
*  初始化返回文件句柄
**/
ENUM_ERROR_CODE InitYamlInstance(const char* yaml, void** pYamlInstance) {
    if (NULL == yaml) {
        std::cout << "ERROR: yaml file path is empty" << std::endl;
        return ERR_PROFILE_INPUT_FILENAME_EMPTY;
    }

    YAML::Node config;
    try {
        config = YAML::LoadFile(yaml);
    } catch (const std::exception &e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return ERR_PROFILE_READ_CONFIG_ERROR;
    }

    YamlInstance* _instance = new YamlInstance();
    if (NULL == _instance) {
        return ERR_NO_FREE_MEMORY;
    }
    _instance->_param->config = config;
    _instance->_param->yaml_file = yaml;
    _instance->_param->bParamIsOk = true;
    *pYamlInstance = static_cast<void*>(_instance);
    return ENUM_OK;
}

/**
*  输入查询的key返回对应的值
**/
ENUM_ERROR_CODE GetYamlString(void* pYamlInstance, const char *AppName, const char *KeyName, char *KeyVal, size_t keyValSize ){
    YamlInstance* _instance = static_cast<YamlInstance*>(pYamlInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
        std::cerr << "GetYamlString: Invalid instance" << std::endl;
        return ERR_INPUT_INSTANCE_INVALID;
    }

    if (NULL == AppName || NULL == KeyName) {
        std::cerr << "AppName or KeyName is empty" << std::endl;
        return ERR_PROFILE_SEARCH_KEYNAME_EMPTY;
    }

    try {
        YAML::Node appNode = _instance->_param->config[AppName];
        if (!appNode || !appNode[KeyName]) {
            std::cerr << "Key not found in YAML" << std::endl;
            return ERR_PROFILE_NOT_FOUND_SEARCH_KEYNAME;
        }

        std::string value = appNode[KeyName].as<std::string>();
        strncpy(KeyVal, value.c_str(), keyValSize);
    } catch (const std::exception &e) {
        std::cerr << "Error retrieving value: " << e.what() << std::endl;
        return ERR_PROFILE_READ_CONFIG_ERROR;
    }

    return ENUM_OK;
}

ENUM_ERROR_CODE SetYamlString(void* pYamlInstance, const char *AppName, const char *KeyName, const char *KeyVal) {
    YamlInstance* _instance = static_cast<YamlInstance*>(pYamlInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
        std::cerr << "SetYamlString: Invalid instance" << std::endl;
        return ERR_INPUT_INSTANCE_INVALID;
    }

    if (AppName == NULL || KeyName == NULL || KeyVal == NULL) {
        std::cerr << "AppName, KeyName, or KeyVal is empty" << std::endl;
        return ERR_PROFILE_SEARCH_KEYNAME_EMPTY;
    }

    try {
        _instance->_param->config[AppName][KeyName] = KeyVal;

        std::ofstream fout(_instance->_param->yaml_file);
        fout << _instance->_param->config;
    } catch (const std::exception &e) {
        std::cerr << "Error writing to YAML file: " << e.what() << std::endl;
        return ERR_PROFILE_CONFIG_WRITE_ERROR;
    }

    return ENUM_OK;
}

ENUM_ERROR_CODE DestorytYamlInstance(void** pYamlInstance) {
    YamlInstance* instance = static_cast<YamlInstance*>(*pYamlInstance);
    if (!instance || !instance->_param->bParamIsOk) {
        std::cerr << "DestorytYamlInstance: Invalid instance" << std::endl;
        return ERR_INPUT_INSTANCE_INVALID;
    }

    delete instance;
    *pYamlInstance = NULL;
    return ENUM_OK;
}