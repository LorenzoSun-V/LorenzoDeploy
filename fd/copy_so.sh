# 目标目录，存放复制的.so文件
DEST_DIR="/lorenzo/deploy/FastDeploy/build/compiled_fastdeploy_sdk/libs"

# 源目录，包含.so文件的目录及其子目录
SRC_DIR="/lorenzo/deploy/FastDeploy/build/compiled_fastdeploy_sdk/third_libs/install"

# 如果目标目录不存在，则创建
if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
fi

# 查找并复制.so文件
find "$SRC_DIR" -type f -name "*.so*" -exec cp {} "$DEST_DIR" \;

echo "所有.so文件已被复制到$DEST_DIR目录下。"