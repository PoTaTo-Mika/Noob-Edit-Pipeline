import json
import os
import glob

def convert_json_to_jsonl(input_pattern, output_path):
    """
    将多个JSON文件合并成JSONL格式，并为每个JSON添加文件名作为pid
    
    Args:
        input_pattern: 输入JSON文件的匹配模式，如 "data/*.json"
        output_path: 输出JSONL文件的路径
    """
    # 获取所有匹配的JSON文件
    json_files = glob.glob(input_pattern)
    
    if not json_files:
        print(f"没有找到匹配的文件: {input_pattern}")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for json_file in json_files:
            try:
                # 读取JSON文件
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 获取文件名（不带扩展名）作为pid
                filename = os.path.splitext(os.path.basename(json_file))[0]
                
                # 添加pid到数据中
                data['pid'] = filename
                
                # 写入JSONL文件
                jsonl_file.write(json.dumps(data) + '\n')
                
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
    
    print(f"转换完成！输出文件: {output_path}")
    print(f"成功处理 {len(json_files)} 个文件")

if __name__ == "__main__":
    # 配置输入输出路径
    input_pattern = "data/*.json"  # 根据您的实际文件位置调整
    output_path = "data/output.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 执行转换
    convert_json_to_jsonl(input_pattern, output_path)