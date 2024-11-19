/*
 * @FilePath: /bt_alg_api/profile/src/profile.h
 * @Description: 读写ini配置文件接口
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-03 17:49:59
 */
#pragma once

#include "common.h"

extern "C"
{
                                                     
   /*
    * @brief 初始化配置文件获得句柄
    *	     
    * @param  profile               输入配置文件
    * @param  pProfileInstance      返回串口实例
    * 
    * @return ENUM_ERROR_CODE       返回错误码
    */
      ENUM_ERROR_CODE InitProfileInstance(
        const char* profile,
        void** pProfileInstance
    );

   /*
    * @brief   从配置文件中获取信息
    *	     
    * @param  pProfileInstance      输入配置文件句柄
    * @param  AppName               输入检索的类别名
    * @param  KeyName               输入检索的关键字
    * @param  KeyVal                返回字符串形式关键字的值
    * @param  keyValSize            输入字符串长度，最大长度1024
    * 
    * @return ENUM_ERROR_CODE       返回错误码
    */
      ENUM_ERROR_CODE GetProfileString(
        void* pProfileInstance, 
        const char *AppName, 
        const char *KeyName, 
        char *KeyVal,
        size_t keyValSize 
    );     

   /*
    * @brief   添加或者修改配置文件中的值
    *	     
    * @param  pProfileInstance      输入配置文件句柄
    * @param  AppName               输入检索的类别名
    * @param  KeyName               输入检索的关键字
    * @param  KeyVal                输入关键字的值
    * 
    * @return ENUM_ERROR_CODE       返回错误码
    */
      ENUM_ERROR_CODE SetProfileString(
        void* pProfileInstance, 
        const char *AppName, 
        const char *KeyName, 
        const char *KeyVal
    );     

   /*
    * @brief 销毁句柄
    *
    * @param   pProfileInstance      需要销毁的句柄 
    *
    * @return  ENUM_ERROR_CODE       返回OK表示成功
    */

    ENUM_ERROR_CODE DestorytProfileInstance(
        void** pProfileInstance
    );                                            
}
