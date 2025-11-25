import os
import sys
import io
import json
import torch
import numpy as np
from PIL import Image
import webdataset as wds
from torch.multiprocessing import Process, Queue, set_start_method
import queue

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from offline_deps.openpose.open_pose import OpenposeDetector

def decode_image_to_numpy(image_data):
    try:
        with Image.open(io.BytesIO(image_data)) as img:
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

    pipeline = wds.WebDataset(tar_files, shardshuffle=False)
    image_extensions = {
        'webp', 'jpg', 'jpeg', 'png', 'bmp', 'gif', 
        'tiff', 'tif', 'ppm', 'pgm', 'pbm', 'pnm'
    }

    for sample in pipeline:
        key = sample.get('__key__', 'unknown')
        image_data = None
        
        for sample_key in sample.keys():
            if sample_key == '__key__':
                continue
            
            if sample_key.lower() in image_extensions:
                try:
                    image_data = decode_image_to_numpy(sample[sample_key])
                    if image_data is not None:
                        break
                except Exception as e:
                    continue
        
        if image_data is not None:
            yield key, image_data
        else:
            available_keys = [k for k in sample.keys() if k != '__key__']
            print(f"警告: 样本 {key} 未找到有效图像数据")

def worker_process(gpu_id, task_queue, result_queue, output_path):
    """每个GPU上运行的工作进程"""
    print(f"[GPU {gpu_id}] 工作进程启动")
    
    # 设置当前进程使用的GPU
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # 加载模型到指定GPU
    model = OpenposeDetector.from_pretrained().to(device)
    print(f"[GPU {gpu_id}] 模型加载完成")
    
    processed = 0
    
    while True:
        try:
            # 从队列获取任务
            task = task_queue.get(timeout=5)
            
            if task is None:  # 结束信号
                print(f"[GPU {gpu_id}] 收到结束信号，已处理 {processed} 张图片")
                break
            
            image_key, image_numpy = task
            json_filepath = os.path.join(output_path, f"{image_key}.json")
            
            # 检查是否已处理
            if os.path.exists(json_filepath):
                result_queue.put(('skipped', image_key))
                continue
            
            try:
                # 处理图片
                pose_img, openpose_dict = model(
                    image_numpy,
                    include_body=True,
                    include_face=False,
                    include_hand=False,
                    output_type="np",
                    detect_resolution=512,
                    image_and_json=True,
                    xinsr_stick_scaling=False
                )
                
                # 保存结果
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(openpose_dict, f, indent=4)
                
                result_queue.put(('success', image_key))
                processed += 1
                
            except Exception as e:
                print(f"[GPU {gpu_id}] 处理 {image_key} 失败: {e}")
                result_queue.put(('error', image_key))
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] 工作进程错误: {e}")
            break

def main():
    TAR_DATA_PATH = '../data'
    OUTPUT_PATH = './outputs/openpose_dicts'
    NUM_GPUS = 4  # GPU数量
    
    TAR_DATA_PATH = os.path.expanduser(TAR_DATA_PATH)
    OUTPUT_PATH = os.path.expanduser(OUTPUT_PATH)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"输入目录: {TAR_DATA_PATH}")
    print(f"输出目录: {OUTPUT_PATH}")
    print(f"使用 {NUM_GPUS} 个GPU进行并行处理")
    
    # 创建任务队列和结果队列
    task_queue = Queue(maxsize=NUM_GPUS * 10)  # 限制队列大小避免内存溢出
    result_queue = Queue()
    
    # 启动工作进程
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = Process(target=worker_process, 
                   args=(gpu_id, task_queue, result_queue, OUTPUT_PATH))
        p.start()
        processes.append(p)
    
    # 主进程：读取数据并分发任务
    data_generator = create_data_pipeline(TAR_DATA_PATH)
    
    if data_generator is None:
        print("数据管道创建失败，程序退出。")
        # 发送结束信号
        for _ in range(NUM_GPUS):
            task_queue.put(None)
        for p in processes:
            p.join()
        sys.exit(1)
    
    print("\n开始处理数据流...")
    
    # 统计变量
    total_submitted = 0
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 分发任务
    import threading
    
    def distribute_tasks():
        nonlocal total_submitted
        for image_key, image_numpy in data_generator:
            task_queue.put((image_key, image_numpy))
            total_submitted += 1
            if total_submitted % 100 == 0:
                print(f"已提交 {total_submitted} 个任务到队列")
        
        # 发送结束信号
        for _ in range(NUM_GPUS):
            task_queue.put(None)
        print(f"所有任务已提交，共 {total_submitted} 个")
    
    # 在单独线程中分发任务
    distributor = threading.Thread(target=distribute_tasks)
    distributor.start()
    
    # 收集结果
    finished_workers = 0
    while finished_workers < NUM_GPUS:
        try:
            result = result_queue.get(timeout=1)
            status, image_key = result
            
            if status == 'success':
                processed_count += 1
            elif status == 'skipped':
                skipped_count += 1
            elif status == 'error':
                error_count += 1
            
            total_done = processed_count + skipped_count + error_count
            if total_done % 50 == 0:
                print(f"\n[进度] 已完成: {total_done}/{total_submitted} | "
                      f"新增: {processed_count} | 跳过: {skipped_count} | 错误: {error_count}")
                
        except queue.Empty:
            # 检查工作进程是否都结束了
            alive_count = sum(1 for p in processes if p.is_alive())
            if alive_count == 0:
                break
            continue
    
    # 等待所有进程结束
    distributor.join()
    for p in processes:
        p.join()
    
    print("\n" + "="*60)
    print("所有 .tar 文件处理完成。")
    print(f"✓ 本次运行新增处理: {processed_count} 张图片")
    print(f"↷ 检测到并跳过已处理图片: {skipped_count} 张")
    print(f"✗ 处理失败: {error_count} 张")
    print(f"总计: {processed_count + skipped_count + error_count} 张")
    print("="*60)

if __name__ == '__main__':
    # 设置多进程启动方式
    set_start_method('spawn', force=True)
    main()