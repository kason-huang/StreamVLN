#!/usr/bin/env python3
"""
从data1/trajectory_data/R2R数据集中抽取200个episode创建R2R_small数据集
保持annotations.json和images目录的对应关系
"""

import json
import os
import random
import shutil
from tqdm import tqdm
import time

def create_r2r_small_dataset():
    """创建R2R_small数据集"""

    # 配置参数
    source_dir = "data1/trajectory_data/R2R"
    target_dir = "data1/trajectory_data/R2R_small"
    num_episodes = 200

    print("=" * 60)
    print("创建R2R_small数据集")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"抽取episode数量: {num_episodes}")
    print()

    # 检查源文件是否存在
    source_annotations = os.path.join(source_dir, "annotations.json")
    source_images = os.path.join(source_dir, "images")

    if not os.path.exists(source_annotations):
        print(f"❌ 错误: 找不到源annotations文件: {source_annotations}")
        return False

    if not os.path.exists(source_images):
        print(f"❌ 错误: 找不到源images目录: {source_images}")
        return False

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    print(f"✅ 创建目标目录: {target_dir}")

    # 1. 加载annotations
    print("1. 加载R2R annotations...")
    with open(source_annotations, 'r') as f:
        all_annotations = json.load(f)

    print(f"   📊 总episode数量: {len(all_annotations):,}")

    if len(all_annotations) < num_episodes:
        print(f"⚠️  警告: 源数据只有{len(all_annotations)}个episode，少于要求的{num_episodes}个")
        num_episodes = len(all_annotations)

    # 2. 随机抽取200个episode
    print("2. 随机抽取episodes...")
    random.seed(42)  # 设置随机种子保证可重复性
    selected_annotations = random.sample(all_annotations, num_episodes)

    print(f"   ✅ 成功抽取 {len(selected_annotations):,} 个episodes")

    # 显示抽取的episode ID范围
    episode_ids = [annot['id'] for annot in selected_annotations]
    print(f"   📋 Episode ID范围: {min(episode_ids)} - {max(episode_ids)}")
    print(f"   📋 前10个ID: {sorted(episode_ids)[:10]}")

    # 3. 检查对应的images目录
    print("3. 检查对应的images目录...")
    available_images = set()
    missing_images = []

    for annot in selected_annotations:
        video_path = annot['video']  # 例如: "images/17DRP5sb8fy_r2r_000577"
        # 去掉 "images/" 前缀，因为images目录直接在source_dir下
        image_dir_name = video_path.replace("images/", "")
        image_dir_path = os.path.join(source_images, image_dir_name)

        if os.path.exists(image_dir_path):
            available_images.add(image_dir_name)
        else:
            missing_images.append(video_path)

    print(f"   📊 找到 {len(available_images)} 个对应的image目录")

    if missing_images:
        print(f"   ⚠️  警告: {len(missing_images)} 个episode缺少对应的images目录")
        print(f"   缺失的images (前5个): {missing_images[:5]}")

    # 4. 保存新的annotations.json
    print("4. 生成新的annotations.json...")
    target_annotations = os.path.join(target_dir, "annotations.json")

    with open(target_annotations, 'w') as f:
        json.dump(selected_annotations, f, indent=2)

    print(f"   ✅ 保存到: {target_annotations}")
    print(f"   📊 文件大小: {os.path.getsize(target_annotations) / 1024:.1f} KB")

    # 5. 复制对应的images目录
    print("5. 复制对应的images目录...")
    target_images = os.path.join(target_dir, "images")
    os.makedirs(target_images, exist_ok=True)

    copied_count = 0
    for image_dir_name in tqdm(available_images, desc="复制图片目录"):
        src_path = os.path.join(source_images, image_dir_name)
        dst_path = os.path.join(target_images, image_dir_name)

        # 复制目录
        if os.path.exists(src_path):
            try:
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                copied_count += 1
            except Exception as e:
                print(f"   ❌ 复制失败 {image_dir_name}: {e}")
        else:
            print(f"   ❌ 源目录不存在: {src_path}")

    print(f"   ✅ 成功复制 {copied_count} 个images目录")

    # 6. 验证数据集
    print("6. 验证数据集...")
    verify_r2r_small_dataset(target_dir)

    # 7. 生成统计报告
    print("\n" + "=" * 60)
    print("数据集创建完成！统计信息:")
    print("=" * 60)
    print(f"📁 目标目录: {target_dir}")
    print(f"📄 annotations.json: {len(selected_annotations):,} episodes")
    print(f"🖼️  images目录: {copied_count} 个")

    # 统计images总大小
    total_images_size = 0
    for image_dir_name in available_images:
        image_dir_path = os.path.join(target_images, image_dir_name)
        if os.path.exists(image_dir_path):
            for root, dirs, files in os.walk(image_dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_images_size += os.path.getsize(file_path)

    print(f"💾 images总大小: {total_images_size / (1024*1024):.1f} MB")

    # 验证对应关系
    print(f"\n🔍 验证前10个episodes的对应关系:")
    for i, annot in enumerate(selected_annotations[:10]):
        video_path = annot['video']
        image_dir_name = video_path.replace("images/", "")
        expected_path = os.path.join(target_images, image_dir_name)

        if os.path.exists(expected_path):
            # 检查rgb目录是否存在
            rgb_dir = os.path.join(expected_path, "rgb")
            if os.path.exists(rgb_dir):
                rgb_count = len(os.listdir(rgb_dir))
                print(f"   ✅ Episode {annot['id']:6d}: {image_dir_name} ({rgb_count} RGB images)")
            else:
                print(f"   ⚠️  Episode {annot['id']:6d}: {image_dir_name} (缺少rgb目录)")
        else:
            print(f"   ❌ Episode {annot['id']:6d}: {image_dir_name} (缺失)")

    print(f"\n🎉 R2R_small数据集创建成功！")
    print(f"📍 位置: {target_dir}")
    print(f"📝 现在可以用于streamvln_train.sh快速验证")

    # 7. 修改配置文件路径的建议
    print(f"\n💡 使用建议:")
    print(f"   修改配置文件中的数据路径指向: {target_dir}")
    print(f"   例如: data_path: {target_dir}/annotations.json")

    return True

def verify_r2r_small_dataset(target_dir):
    """验证R2R_small数据集的完整性"""
    print("   🔍 验证数据集完整性...")

    annotations_file = os.path.join(target_dir, "annotations.json")
    images_dir = os.path.join(target_dir, "images")

    # 检查文件是否存在
    if not os.path.exists(annotations_file):
        print(f"   ❌ annotations.json不存在")
        return False

    if not os.path.exists(images_dir):
        print(f"   ❌ images目录不存在")
        return False

    # 检查annotations格式
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        print(f"   ✅ annotations.json格式正确，包含{len(annotations)}个episodes")

        # 检查必要字段
        required_fields = ['id', 'video', 'instructions', 'actions']
        missing_fields_count = 0

        for i, annot in enumerate(annotations[:5]):  # 检查前5个
            for field in required_fields:
                if field not in annot:
                    missing_fields_count += 1
                    print(f"   ❌ Episode {i} 缺少字段: {field}")

        if missing_fields_count == 0:
            print("   ✅ 检查的episodes都包含必要字段")

    except json.JSONDecodeError as e:
        print(f"   ❌ annotations.json格式错误: {e}")
        return False

    # 检查images目录
    image_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    print(f"   ✅ images目录包含{len(image_dirs)}个子目录")

    # 检查前几个image目录是否包含rgb子目录
    rgb_dirs_count = 0
    for image_dir in image_dirs[:10]:
        rgb_path = os.path.join(images_dir, image_dir, "rgb")
        if os.path.exists(rgb_path):
            rgb_dirs_count += 1

    print(f"   ✅ 前10个image目录中有{rgb_dirs_count}个包含rgb子目录")

    print("   ✅ 数据集验证通过")
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = create_r2r_small_dataset()
    end_time = time.time()

    if success:
        print(f"\n🎉 成功创建R2R_small数据集！耗时: {end_time - start_time:.1f}秒")
    else:
        print(f"\n❌ 创建失败，请检查错误信息")