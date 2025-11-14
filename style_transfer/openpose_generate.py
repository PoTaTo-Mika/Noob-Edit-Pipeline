import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import io
import json
import torch
import numpy as np
from PIL import Image
import webdataset as wds
from offline_deps.openpose.open_pose import OpenposeDetector

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

print("正在加载 OpenPose 模型...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = OpenposeDetector.from_pretrained().to(device)
print(f"模型加载完成，运行在 {device} 设备上。")

def decode_webp_to_numpy(webp_data):
    try:
        with Image.open(io.BytesIO(webp_data)) as img:
            return np.array(img.convert("RGB"))
    except Exception as e:
        print(f"警告: 图片解码失败 - {e}")
        return None

def create_data_pipeline(tar_directory_path):
    try:
        tar_files = [
            os.path.join(tar_directory_path, f)
            for f in os.listdir(tar_directory_path)
            if f.endswith(".tar")
        ]
    except FileNotFoundError:
        print(f"错误: 目录 '{tar_directory_path}' 不存在。")
        return None

    if not tar_files:
        print(f"警告: 在目录 '{tar_directory_path}' 中没有找到 .tar 文件。")
        return None

    print(f"在 '{tar_directory_path}' 中发现 {len(tar_files)} 个 .tar 文件。")
    print("文件列表:")
    for i, tar_file in enumerate(tar_files, 1):
        print(f"  {i}. {os.path.basename(tar_file)}")

    pipeline = wds.WebDataset(tar_files).map_dict(webp=decode_webp_to_numpy)

    for sample in pipeline:
        if sample.get('webp') is not None:
            yield sample['__key__'], sample['webp']
        else:
            print(f"警告: 样本 {sample.get('__key__', 'unknown')} 解码失败，跳过处理。")

def process_one_pic(image_np, openpose_model):
    try:
        print(f"正在处理图片，形状: {image_np.shape}")
        pose_img, openpose_dict = openpose_model(
            image_np,
            include_body=True,
            include_face=False,
            include_hand=False,
            output_type="np",
            detect_resolution=512,
            image_and_json=True,
            xinsr_stick_scaling=False
        )
        print(f"姿势检测完成，检测到 {len(openpose_dict.get('people', []))} 个人物")
        return pose_img, openpose_dict
    except Exception as e:
        print(f"错误: 姿势检测失败 - {e}")
        return None, None

if __name__ == '__main__':

    TAR_DATA_PATH = './data'
    OUTPUT_PATH = './outputs'

    TAR_DATA_PATH = os.path.expanduser(TAR_DATA_PATH)
    OUTPUT_PATH = os.path.expanduser(OUTPUT_PATH)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"输入目录: {TAR_DATA_PATH}")
    print(f"输出目录: {OUTPUT_PATH}")

    data_generator = create_data_pipeline(TAR_DATA_PATH)

    if data_generator is None:
        print("数据管道创建失败，程序退出。")
        sys.exit(1)
    else:
        print("\n开始处理数据流（将自动跳过已处理的文件）...")
        processed_count = 0
        skipped_count = 0
        error_count = 0

        for image_key, image_numpy in data_generator:
            print(f"\n--- 处理图片: {image_key} ---")

            json_filepath = os.path.join(OUTPUT_PATH, f"{image_key}.json")
            
            if os.path.exists(json_filepath):
                print(f"跳过已处理文件: {image_key}")
                skipped_count += 1
                continue
            
            pose_image_result, pose_dict_result = process_one_pic(image_numpy, model)

            if pose_dict_result is None:
                print(f"错误: {image_key} 姿势检测失败，跳过此文件")
                error_count += 1
                continue

            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(pose_dict_result, f, indent=4)
                print(f"✓ 成功保存姿势数据: {image_key}.json")
                processed_count += 1
            except Exception as e:
                print(f"错误: 保存文件失败 - {e}")
                error_count += 1

            if processed_count % 10 == 0:
                print(f"\n[进度更新] 已处理: {processed_count} | 跳过: {skipped_count} | 错误: {error_count}")

        print("\n" + "="*60)
        print("所有 .tar 文件处理完成。")
        print(f"✓ 本次运行新增处理: {processed_count} 张图片")
        print(f"↷ 检测到并跳过已处理图片: {skipped_count} 张")
        print(f"✗ 处理失败: {error_count} 张")
        print(f"总计处理文件: {processed_count + skipped_count + error_count} 张")
        print("="*60)