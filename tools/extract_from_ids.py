import os
import sys
import io
import json
import webdataset as wds
from PIL import Image
import argparse

def setup_paths():
    """设置项目路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    return parent_dir

def load_target_ids(txt_file_path):
    """从txt文件加载目标ID列表"""
    target_ids = set()
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    target_ids.add(line)
        print(f"成功加载 {len(target_ids)} 个目标ID")
        return target_ids
    except Exception as e:
        print(f"错误: 无法读取ID文件 - {e}")
        return None

def create_data_pipeline(tar_directory_path):
    """创建WebDataset数据管道"""
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
    return wds.WebDataset(tar_files, shardshuffle=False)

def extract_images_by_ids(tar_directory, ids_file, output_dir):
    """根据ID列表从tar文件中提取图片"""
    # 加载目标ID
    target_ids = load_target_ids(ids_file)
    if target_ids is None:
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据管道
    pipeline = create_data_pipeline(tar_directory)
    if pipeline is None:
        return
    
    image_extensions = {
        'webp', 'jpg', 'jpeg', 'png', 'bmp', 'gif', 
        'tiff', 'tif', 'ppm', 'pgm', 'pbm', 'pnm'
    }
    
    extracted_count = 0
    missing_ids = set(target_ids)
    
    print("开始提取图片...")
    
    for sample in pipeline:
        key = sample.get('__key__', 'unknown')
        
        # 检查是否是目标ID
        if key not in target_ids:
            continue
            
        # 查找图像数据
        image_data = None
        image_ext = None
        
        for sample_key in sample.keys():
            if sample_key == '__key__':
                continue
            
            if sample_key.lower() in image_extensions:
                try:
                    image_data = sample[sample_key]
                    image_ext = sample_key.lower()
                    break
                except Exception as e:
                    continue
        
        if image_data is not None:
            # 保存图片
            output_path = os.path.join(output_dir, f"{key}.{image_ext}")
            try:
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                extracted_count += 1
                missing_ids.remove(key)
                print(f"已提取: {key} -> {output_path}")
                
                # 如果所有目标都已找到，提前退出
                if not missing_ids:
                    break
                    
            except Exception as e:
                print(f"错误: 保存图片 {key} 失败 - {e}")
        else:
            print(f"警告: 样本 {key} 未找到有效图像数据")
            if key in missing_ids:
                missing_ids.remove(key)
    
    print(f"\n提取完成!")
    print(f"成功提取: {extracted_count} 张图片")
    print(f"未找到的ID: {len(missing_ids)} 个")
    
    if missing_ids:
        print("未找到的ID列表:")
        for missing_id in sorted(missing_ids):
            print(f"  - {missing_id}")
        
        # 保存未找到的ID到文件
        missing_file = os.path.join(output_dir, "missing_ids.txt")
        with open(missing_file, 'w', encoding='utf-8') as f:
            for missing_id in sorted(missing_ids):
                f.write(f"{missing_id}\n")
        print(f"未找到的ID已保存到: {missing_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='根据ID列表从tar文件中提取图片')
    parser.add_argument('--tar_dir', required=True, help='包含tar文件的目录路径')
    parser.add_argument('--ids_file', required=True, help='包含ID列表的txt文件路径')
    parser.add_argument('--output_dir', default='data/extracted_images', help='输出目录路径')
    
    args = parser.parse_args()
    
    # 设置路径
    setup_paths()
    
    # 提取图片
    extract_images_by_ids(args.tar_dir, args.ids_file, args.output_dir)

if __name__ == "__main__":
    main()