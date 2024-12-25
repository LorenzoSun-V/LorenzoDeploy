/*
 * @FilePath: /bt_alg_api/profile/src/profile.cpp
 * @Description: 读写ini配置文件接口
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-06 11:28:56
 */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <memory>
#include "profile.h"


#define MAX_VALUE 1024   /* Define the maximum length of a section or key */
#define TRIM_LEFT  1   /* Remove left whitespace */
#define TRIM_RIGHT 2   /* Remove right whitespace */
#define TRIM_SPACE 3   /* Remove both left and right whitespace */

typedef struct _option
{
    char key[MAX_VALUE];   /* Key */
    char value[MAX_VALUE]; /* Value */
    struct _option* next;  /* Linked list connection */
} Option;

typedef struct _data
{
    char section[MAX_VALUE]; /* Section */
    Option* option;         /* Option linked list head */
    struct _data* next;     /* Linked list connection */
} Data;

typedef struct
{
    char comment;              /* Comment character */
    char separator;            /* Separator character */
    char re_string[MAX_VALUE]; /* String result */
    int re_int;               /* Integer result */
    bool re_bool;              /* Boolean result */
    double re_double;         /* Double result */
    Data* data;               /* Data */
} Config;


class ReadConfParam {
public:
    bool bParamIsOk;
    const char *profile;

    ReadConfParam() 
    {
        bParamIsOk= false;
    }

};

class ReadConfInstance {
public:
   std::shared_ptr<ReadConfParam> _param;

   ReadConfInstance() {
        _param = std::make_shared<ReadConfParam>();
    }
};

/**
* 判断字符串是否为空
* 为空返回true,不为空返回false
**/
bool str_empty(const char* string)
{
    return NULL == string || '\0' == *string;
}

/**
* 向链表添加section,key,value
* 如果添加时不存在section则新增一个
* 如果对应section的key不存在则新增一个
* 如果section已存在则不会重复创建
* 如果对应section的key已存在则只会覆盖key的值
**/
bool cnf_add_option(Config* cnf, const char* section, const char* key, const char* value)
{
if (NULL == cnf || str_empty(section) || str_empty(key) || str_empty(value))
    {
        /* 参数不正确,返回false */
        return false;
    }
 
    /* 让变量p循环遍历data,找到对应section */
    Data* p = cnf->data;
    while (NULL != p && 0 != strcmp(p->section, section))
    {
        p = p->next;
    }
 
    if (NULL == p)
    {
        /* 说明没有找到section,需要加一个 */
        Data* ps = (Data*)malloc(sizeof(Data));
        if (NULL == ps)
        {
            /* 申请内存错误 */
            exit(-1);
        }
        strcpy(ps->section, section);
        ps->option = NULL;    /* 初始的option要为空 */
        ps->next = cnf->data; /* cnf->data可能为NULL */
        cnf->data = p = ps;   /* 头插法插入链表 */
    }
 
    Option* q = p->option;
    while (NULL != q && 0 != strcmp(q->key, key))
    {
        /* 遍历option,检查key是否已经存在 */
        q = q->next;
    }
 
    if (NULL == q)
    {
        /* 不存在option,则新建一个 */
        q = (Option*)malloc(sizeof(Option));
        if (NULL == q)
        {
            /* 申请内存错误 */
            exit(-1);
        }
        strcpy(q->key, key);
        q->next = p->option;     /* 这里p->option可能为NULL,不过也没关系 */
        p->option = q;           /* 头插法插入链表 */
    }
    strcpy(q->value, value);     /* 无论如何要把值改了 */
 
    return true;
}

/**
* 按照参数去除字符串左右空白
**/
char* trim_string(char* string, int mode)
{
    char* left = string;
    if ((mode & 1) != 0)
    {
        // 去除左边空白字符
        for (; *left != '\0'; left++)
        {
            if (0 == isspace(*left))
            {
                break;
            }
        }
    }
    if ((mode & 2) != 0)
    {
        // 去除右边空白字符
        char* right = string - 1 + strlen(string);
        for (; right >= left; right--)
        {
            if (0 == isspace(*right))
            {
                *(right + 1) = '\0';
                break;
            }
        }
    }
    return left;
}

/**
* 传递配置文件路径
* 参数有文件路径,注释字符,分隔符
* 返回Config结构体
**/
Config* cnf_read_config(const char* filename, char comment, char separator)
{
    Config* cnf = (Config*)malloc(sizeof(Config));
    if (NULL == cnf)
    {
        /* 申请内存错误 */
        exit(-1);
    }
    cnf->comment = comment;       /* 每一行该字符及以后的字符将丢弃 */
    cnf->separator = separator;   /* 用来分隔Section 和 数据 */
    cnf->data = NULL;             /* 初始数据为空 */
 
    if (str_empty(filename))
    {
        /* 空字符串则直接返回对象 */
        return cnf;
    }
 
    FILE* fp = fopen(filename, "r");
    if (NULL == fp)
    {
        /* 读文件错误直接按照错误码退出 */
        exit(errno);
    }
 
    /* 保存一行数据到字符串 */
    char* s, * e, * pLine, sLine[MAX_VALUE];
 
    /* 缓存section,key,value */
    char section[MAX_VALUE] = { '\0' }, key[MAX_VALUE], value[MAX_VALUE];
    while (NULL != fgets(sLine, MAX_VALUE, fp))
    {
        /* 去掉一行两边的空白字符 */
        pLine = trim_string(sLine, TRIM_SPACE);
        if (*pLine == '\0' || *pLine == comment)
        {
            /* 空行或注释行跳过 */
            continue;
        }
        s = strchr(pLine, comment);
        if (s != NULL)
        {
            /* 忽略本行注释后的字符 */
            *s = '\0';
        }
 
        s = strchr(pLine, '[');
        if (s != NULL) {
            e = strchr(++s, ']');
            if (e != NULL)
            {
                /* 找到section */
                *e = '\0';
                strcpy(section, s);
            }
        }
        else
        {
            s = strchr(pLine, separator);
 
            /* 找到包含separator的行,且前面行已经找到section */
            if (s != NULL && *section != '\0')
            {
                /* 将分隔符前后分成2个字符串 */
                *s = '\0';
                strcpy(key, trim_string(pLine, TRIM_RIGHT));   /* 赋值key */
                strcpy(value, trim_string(s + 1, TRIM_LEFT));  /* 赋值value */
                cnf_add_option(cnf, section, key, value);      /* 添加section,key,value */
            }
        }
    } /* end while */
    fclose(fp);
    return cnf;
}

/**
* 获取指定类型的值
* 根据不同类型会赋值给对应值
* 本方法需要注意,int和double的转换,不满足就是0
**/
bool cnf_get_value(Config* cnf, const char* section, const char* key)
{
/* 让变量p循环遍历data,找到对应section */
    Data* p = cnf->data;
    while (NULL != p && 0 != strcmp(p->section, section))
    {
        p = p->next;
    }
 
    /* 节点Section没有找到 */
    if (NULL == p)
    {
        return false;
    }
 
    Option* q = p->option;
    while (NULL != q && 0 != strcmp(q->key, key))
    {
        /* 遍历option,检查key是否已经存在 */
        q = q->next;
    }
 
    /*节点key没有找到*/
    if (NULL == q)
    {
        return false;
    }
 
    strcpy(cnf->re_string, q->value);                   /* 将结果字符串赋值 */
    cnf->re_int = atoi(cnf->re_string);                 /* 转换为整形 */
    cnf->re_bool = 0 == strcmp("true", cnf->re_string); /* 转换为bool型 */
    cnf->re_double = atof(cnf->re_string);              /* 转换为double型 */
 
    return true;
}

/**
* 判断section是否存在
* 不存在返回空指针
* 存在则返回包含那个section的Data指针
**/
Data* cnf_has_section(Config* cnf, const char* section)
{
 /* 让变量p循环遍历data,找到对应section */
    Data* p = cnf->data;
    while (NULL != p && 0 != strcmp(p->section, section))
    {
        p = p->next;
    }
 
    if (NULL == p)
    {
        /* 没找到则不存在 */
        return NULL;
    }
 
    return p;
}

/**
* 判断指定option是否存在
* 不存在返回空指针
* 存在则返回包含那个section下key的Option指针
**/
Option* cnf_has_option(Config* cnf, const char* section, const char* key)
{
  Data* p = cnf_has_section(cnf, section);
    if (NULL == p)
    {
        /* 没找到则不存在 */
        return NULL;
    }
 
    Option* q = p->option;
    while (NULL != q && 0 != strcmp(q->key, key))
    {
        /* 遍历option,检查key是否已经存在 */
        q = q->next;
    }
    if (NULL == q)
    {
        /* 没找到则不存在 */
        return NULL;
    }
 
    return q;
}


/**
* 将Config对象写入指定文件中
* header表示在文件开头加一句注释
* 写入成功则返回true
**/
bool cnf_write_file(Config* cnf, const char* filename, const char* header)
{
    FILE* fp = fopen(filename, "w");
    if (NULL == fp)
    {
        /* 读文件错误直接按照错误码退出 */
        exit(errno);
    }
 
    if (!str_empty(header))
    {
        /* 文件注释不为空,则写注释到文件 */
        fprintf(fp, "%c %s\n\n", cnf->comment, header);
    }
 
    Option* q;
    Data* p = cnf->data;
    while (NULL != p)
    {
        fprintf(fp, "[%s]\n", p->section);
        q = p->option;
        while (NULL != q)
        {
            fprintf(fp, "%s %c %s\n", q->key, cnf->separator, q->value);
            q = q->next;
        }
        p = p->next;
    }
    fclose(fp);
    return true;
}

/**
* 删除option
**/
bool cnf_remove_option(Config* cnf, const char* section, const char* key)
{
Data* ps = cnf_has_section(cnf, section);
    if (NULL == ps)
    {
        /* 没找到则不存在 */
        return false;
    }
 
    Option* p, * q;
    q = p = ps->option;
    while (NULL != p && 0 != strcmp(p->key, key))
    {
        if (p != q)
        {
            q = q->next;
        }
        /* 始终让q处于p的上一个节点 */
        p = p->next;
    }
 
    if (NULL == p)
    {
        /* 没找到则不存在 */
        return false;
    }
 
    if (p == q)
    {
        /* 第一个option就匹配了 */
        ps->option = p->next;
    }
    else
    {
        q->next = p->next;
    }
 
    free(p);
    q = p = NULL;
    return true;
}

 
/**
* 删除section
**/
bool cnf_remove_section(Config* cnf, const char* section)
{
if (str_empty(section))
    {
        return false;
    }
 
    Data* p, * q;
 
    /* 让变量p循环遍历data,找到对应section */
    q = p = cnf->data;
    while (NULL != p && 0 != strcmp(p->section, section))
    {
        if (p != q)
        {
            /* 始终让q处于p的上一个节点 */
            q = q->next;
        }
        p = p->next;
    }
 
    if (NULL == p)
    {
        /* 没有找到section */
        return false;
    }
 
    if (p == q)
    {
        /* 这里表示第一个section,因此链表头位置改变 */
        cnf->data = p->next;
    }
    else
    {
        /* 此时是中间或尾部节点 */
        q->next = p->next;
    }
 
    Option* ot, * o = p->option;
    while (NULL != o)
    {
        /* 循环释放所有option */
        ot = o;
        o = o->next;
        free(ot);
    }
    p->option = NULL;
    free(p);
    q = p = NULL;
    return true;
}

/**
* 销毁Config对象
* 删除所有数据
**/
void destroy_config(Config** cnf)
{
  if (NULL != *cnf)
    {
        if (NULL != (*cnf)->data)
        {
            Data* pt, * p = (*cnf)->data;
            Option* qt, * q;
            while (NULL != p)
            {
                q = p->option;
                while (NULL != q)
                {
                    qt = q;
                    q = q->next;
                    free(qt);
                }
                pt = p;
                p = p->next;
                free(pt);
            }
        }
        free(*cnf);
        *cnf = NULL;
    }
}

/**
* 打印当前Config对象
**/
void print_config(Config* cnf)
{

    Data* p = cnf->data;
    while (NULL != p)
    {
        printf("[%s]\n", p->section);
        Option* q = p->option;
        while (NULL != q)
        {
            printf("%s%c%s\n", q->key, cnf->separator, q->value);
            q = q->next;
        }
        p = p->next;
    }
}

/**
*  初始化返回文件句柄
**/
ENUM_ERROR_CODE InitProfileInstance(const char* profile,  void** pProfileInstance)
{
    if(NULL == profile) 
    {
        std::cout << "ERROR: pProfilePath is empty" <<std::endl;
       	return ERR_PROFILE_INPUT_FILENAME_EMPTY;
    }

    ReadConfInstance *_instance = new ReadConfInstance();
    if (NULL == _instance) {
        return ERR_NO_FREE_MEMORY;
    }
    _instance->_param->profile = profile;
    _instance->_param->bParamIsOk = true; 
    *pProfileInstance = (void*)_instance;
    return ENUM_OK;

}

/**
*  输入查询的key返回对应的值
**/
ENUM_ERROR_CODE GetProfileString(void* pProfileInstance, const char *AppName, const char *KeyName, char *KeyVal, size_t keyValSize)
{
    ReadConfInstance* _instance = static_cast<ReadConfInstance*>(pProfileInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
	    perror("GetProfileString pProfileInstance is NULL");
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if(NULL == AppName || NULL == KeyName) {
        std::cout << "AppName and KeyName is empty" <<std::endl;
        return ERR_PROFILE_SEARCH_KEYNAME_EMPTY;
    }
    //打开配置文件
    Config* cnf = cnf_read_config( _instance->_param->profile, '#', '=');
    if (NULL == cnf)
    {
        std::cout << "cnf_read_config error" <<std::endl;
        return ERR_PROFILE_READ_CONFIG_ERROR;
    }

    bool bRet = cnf_get_value(cnf, AppName, KeyName);
    if( !bRet ){
        std::cout << "cnf_get_value can't search keynam" <<std::endl;
        return ERR_PROFILE_NOT_FOUND_SEARCH_KEYNAME;
    }
    strncpy(KeyVal, cnf->re_string, keyValSize);
    destroy_config(&cnf);
    return ENUM_OK;

}

ENUM_ERROR_CODE SetProfileString(void* pProfileInstance,  const char *AppName, const char *KeyName, const char *KeyVal)
{
    ReadConfInstance* _instance = static_cast<ReadConfInstance*>(pProfileInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
	     perror("GetProfileString instance is NULL");
	    return ERR_INPUT_INSTANCE_INVALID;
    }

    if(NULL == AppName || NULL == KeyName || NULL == KeyVal) {
        std::cout << "AppName and KeyName is empty" <<std::endl;
        return ERR_INVALID_PARAM;
    }
    //打开配置文件
    Config* cnf = cnf_read_config( _instance->_param->profile, '#', '=');
    if (NULL == cnf)
    {
        std::cout << "cnf_read_config error" <<std::endl;
        return ERR_PROFILE_READ_CONFIG_ERROR; 
    }
    //新增字段
    bool set_result = cnf_add_option(cnf, AppName, KeyName, KeyVal);
    if(!set_result)
    {
        std::cout << "keyval hasn't  been addition." << std::endl;
        destroy_config(&cnf);
        return ERR_PROFILE_CONFIG_ADD_OPTION_ERROR;
    }

    // 写入文件并保存
    bool file_result = cnf_write_file(cnf,  _instance->_param->profile, "BOTON Configuration File");
    if (!file_result)
    {
        std::cout << "File hasn't been written." << std::endl;
        return ERR_PROFILE_CONFIG_WRITE_ERROR;
    }
    destroy_config(&cnf);
    return ENUM_OK;
}


ENUM_ERROR_CODE DestorytProfileInstance( void **pProfileInstance )
{
    ReadConfInstance* _instance = static_cast<ReadConfInstance*>(*pProfileInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "DestorytProfileInstance pProfileInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    //销毁句柄文件
    if (NULL != _instance) {
        delete _instance;
        *pProfileInstance = NULL;
    }            
    return ENUM_OK;
}