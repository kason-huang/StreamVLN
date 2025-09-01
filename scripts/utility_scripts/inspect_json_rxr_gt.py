import gzip
import json
import os
import sys

def check_json_gz_structure(file_path):
    """
    解压 .json.gz 文件并打印其内容结构。
    
    Args:
        file_path (str): .json.gz 文件的路径。
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。")
        return

    print(f"正在读取并解压文件: {file_path}")

    try:
        with gzip.open(file_path, 'rb') as f_in:
            data = json.load(f_in)
        
        print("\n文件成功加载。以下是其数据结构:")

        # 检查根数据类型
        if isinstance(data, list):
            print(f"根数据类型: list，总共有 {len(data)} 个元素。")
            if len(data) > 0:
                print("\n--- 列表前5个元素的结构示例 ---")
                for i, item in enumerate(data[:5]):
                    print(f"\n元素 {i+1} 的数据类型: {type(item)}")
                    if isinstance(item, dict):
                        print("  - 键 (keys):", ", ".join(item.keys()))

        elif isinstance(data, dict):
            print(f"根数据类型: dict，总共有 {len(data)} 个键。")
            if len(data) > 0:
                first_key = next(iter(data))
                first_value = data[first_key]
                
                print(f"\n--- 第一个键值对示例 ---")
                print(f"键: '{first_key}'")
                print(f"值的数据类型: {type(first_value)}")

                # 如果值是字典，则进一步展示其所有内部键和值
                if isinstance(first_value, dict):
                    print(f"\n--- 内部字典结构 (总共 {len(first_value)} 个键) ---")
                    for inner_key, inner_value in first_value.items():
                        print(f"  - 内部键: '{inner_key}'")
                        print(f"    值类型: {type(inner_value)}")
                        
                        # 限制值的输出长度
                        if isinstance(inner_value, (list, dict)):
                            value_str = json.dumps(inner_value, indent=2, ensure_ascii=False, default=str)
                            lines = value_str.splitlines()[:5]
                            print("    值内容 (截断):")
                            for line in lines:
                                print(f"      {line.strip()}")
                            if len(value_str) > 200:
                                print("      ...")
                        else:
                            print(f"    值内容 (截断): {str(inner_value)[:100]}")

        else:
            print(f"根数据类型: {type(data)}")
            print(f"根数据内容 (截断): {str(data)[:200] + '...'}")

    except (gzip.BadGzipFile, json.JSONDecodeError) as e:
        print(f"错误: 文件处理失败。请检查文件是否为有效的 gzip 和 JSON 格式。详细信息: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 如何使用 ---
# 替换成你的文件路径
file_path = "/shared_space/jiangjiajun/data/streamvln_datasets/datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz"

# 调用函数来检查文件结构
check_json_gz_structure(file_path)