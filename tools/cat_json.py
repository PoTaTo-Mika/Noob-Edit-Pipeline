import json
import os
import glob

def json_to_jsonl(input_dir, output_file):
    """
    将目录下的所有JSON文件合并成JSONL格式
    并为每个JSON添加文件名作为pid
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出JSONL文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 查找目录下所有的json文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"在目录 {input_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as infile:
                    # 读取JSON数据
                    data = json.load(infile)
                    
                    # 获取文件名（不带扩展名）作为pid
                    filename = os.path.splitext(os.path.basename(json_file))[0]
                    
                    # 添加pid到数据中
                    data['pid'] = filename
                    
                    # 写入JSONL文件（每行一个JSON对象）
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
    
    print(f"成功处理 {len(json_files)} 个文件，输出到: {output_file}")

if __name__ == "__main__":
    # 使用示例
    input_directory = "data/raw"  # 修改为您的输入目录
    output_path = "data/output.jsonl"
    
    json_to_jsonl(input_directory, output_path)