<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-11-07 10:43:03
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-11-07 10:45:12
 * @Description: 
-->
## ini文件读取接口

### 1. 编译profile库
```
cd operation_file/profile
sh build.sh
```

### 2. 编译test-profile
```
cd operation_file/test-profile
sh build.sh
```

### 3. 测试
```
cd operation_file/test-profile/build/test
./testprofile operation_file/test-profile/conf/conf.ini
```


## yaml文件读取接口

### 1. 编译yaml库
```
cd operation_file/yaml
sh build.sh
```

### 2. 编译test-yaml
```
cd operation_file/test-yaml
sh build.sh
```

### 3. 测试
```
cd operation_file/test-yaml/build/test
./testprofile operation_file/test-yaml/conf/conf.yaml
```
