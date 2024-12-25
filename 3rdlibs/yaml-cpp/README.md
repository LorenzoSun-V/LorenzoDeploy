## 如何编译 yaml-cpp

1. 创建并进入build目录
```
cd 3rdlibs/yaml-cpp/source_code
mkdir build
cd build
```

2. 使用camke进行配置
```
cmake .. -DCMAKE_INSTALL_PREFIX=${/path/to/install} -DYAML_BUILD_SHARED_LIBS=ON
```

- DCMAKE_INSTALL_PREFIX: 编译完后的安装路径
- DYAML_BUILD_SHARED_LIBS：默认是编译静态库，这个开关是编译动态库

3. 编译项目
```
make -j4
```

4. 安装到指定路径：
```
make install
```

这样，yaml-cpp 就会被安装到你指定的路径 /path/to/install 下。