/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-11-06 13:23:56
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-11-06 14:54:51
 * @Description: yaml文件读取接口
 */
#pragma once

#include "common.h"

extern "C"
{
                                                     
   /*
    * @brief 初始化配置文件获得句柄
    *	     
    * @param  yaml               输入yaml配置文件
    * @param  pYamlInstance      返回串口实例
    * 
    * @return ENUM_ERROR_CODE       返回错误码
    */
      ENUM_ERROR_CODE InitYamlInstance(
        const char* yaml,
        void** pYamlInstance
    );

   /*
    * @brief   从配置文件中获取信息
    *	     
    * @param  pYamlInstance         输入配置文件句柄
    * @param  AppName               输入检索的类别名
    * @param  KeyName               输入检索的关键字
    * @param  KeyVal                返回字符串形式关键字的值
    * @param  keyValSize            输入字符串长度，最大长度1024
    * 
    * @return ENUM_ERROR_CODE       返回错误码
    */
      ENUM_ERROR_CODE GetYamlString(
        void* pYamlInstance, 
        const char *AppName, 
        const char *KeyName, 
        char *KeyVal,
        size_t keyValSize 
    );     

   /*
    * @brief   添加或者修改配置文件中的值
    *	     
    * @param  pYamlInstance      输入配置文件句柄
    * @param  AppName               输入检索的类别名
    * @param  KeyName               输入检索的关键字
    * @param  KeyVal                输入关键字的值
    * 
    * @return ENUM_ERROR_CODE       返回错误码
    */
      ENUM_ERROR_CODE SetYamlString(
        void* pYamlInstance, 
        const char *AppName, 
        const char *KeyName, 
        const char *KeyVal
    );     

   /*
    * @brief 销毁句柄
    *
    * @param   pYamlInstance      需要销毁的句柄 
    *
    * @return  ENUM_ERROR_CODE       返回OK表示成功
    */

    ENUM_ERROR_CODE DestorytYamlInstance(
        void** pYamlInstance
    );                                            
}