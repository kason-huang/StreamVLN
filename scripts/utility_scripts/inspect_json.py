import gzip
import json
import pprint  # 使用 pprint 模块让输出更美观、易读

# --- 配置 ---
# 根据您的第二张图，文件名是 test_challenge_guide.json.gz
file_path = '/shared_space/jiangjiajun/data/streamvln_datasets/datasets/rxr/val_unseen/val_unseen_guide.json.gz'

def read_gzipped_json(path):
    """
    读取并解析 .json.gz 文件。

    Args:
        path (str): 文件路径。

    Returns:
        dict or list: 解析后的Python对象 (字典或列表)。
        None: 如果文件不存在或解析失败。
    """
    try:
        # 使用 gzip.open 以文本模式('rt')打开文件，它会自动处理解压缩
        # encoding='utf-8' 是处理JSON时常用的编码格式
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            # 使用 json.load 从文件对象中读取并解析JSON数据
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"错误：文件 '{path}' 未找到。请确保脚本和文件在同一目录下，或者使用完整路径。")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 '{path}' 内容不是有效的JSON格式。")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None

# --- 主程序 ---
if __name__ == "__main__":
    # 读取数据
    dataset = read_gzipped_json(file_path)

    # 如果成功读取到数据，则进行分析和展示
    if dataset:
        print(f"成功从 '{file_path}' 文件中加载数据！\n")

        # 检查数据顶层结构 (根据您的截图，应该是一个字典)
        if isinstance(dataset, dict):
            print("--- 数据顶层结构 (Keys) ---")
            print(list(dataset.keys()))
            print("-" * 30 + "\n")

            # 根据您的第一张图，'episodes' 是主要的数据列表
            if 'episodes' in dataset:
                episodes_list = dataset['episodes']
                print(f"数据集中包含 {len(episodes_list)} 个 'episodes' (条目)。\n")
                
                # 为了不让输出信息过载，我们只详细打印第一个 episode 的结构
                if len(episodes_list) > 0:
                    print("--- 第一个 'episode' 的数据结构示例 ---")
                    # 使用 pprint 美化输出，使其更像您截图中的格式
                    pprint.pprint(episodes_list[0])
                else:
                    print("'episodes' 列表为空。")
        else:
            # 如果顶层不是字典，则按实际情况处理
            print("--- 数据结构预览 ---")
            print("数据顶层是一个列表。")
            if len(dataset) > 0:
                 print(f"该列表包含 {len(dataset)} 个元素。")
                 print("--- 第一个元素的结构示例 ---")
                 pprint.pprint(dataset[0])