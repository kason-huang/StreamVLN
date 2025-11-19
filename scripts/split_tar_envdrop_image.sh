# 设置目标目录
TARGET_DIR="./data1/trajectory_data/EnvDrop/images"

# 获取所有子目录并保存到临时文件
find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d > all_subdirs.txt

# 计算需要切分的块数
total_subdirs=$(wc -l < all_subdirs.txt)
split_count=$((total_subdirs / 10))
split_count=$((split_count + (total_subdirs % 10 > 0 ? 1 : 0)))  # 确保切分成至少10个块

# 使用 split 命令将子目录切分成10等份
split -l $split_count all_subdirs.txt subdirs_part_

# 对每个部分进行 tar.gz 压缩
for part in subdirs_part_*; do
    # 获取当前部分的文件名
    part_name=$(basename "$part")
    
    # 创建压缩文件名
    tar_filename="./data1/trajectory_data/EnvDrop/${TARGET_DIR}_part_${part_name}.tar.gz"
    
    # 将该部分的子目录压缩成 tar.gz
    echo 'please execute: tar -czf "$tar_filename" -T "$part"'
    
    echo "压缩完成：$tar_filename"
done

# 清理临时文件
#rm all_subdirs.txt