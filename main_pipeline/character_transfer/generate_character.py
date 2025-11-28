import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from PIL import Image
from main_pipeline.character_transfer.tag_classify import tag_classify
import json
# 注意：这里导入的是刚才修改的 draw_pose_from_data
from tools.pose_draw import draw_pose_from_data 
import os
import random

# ==========================================
# 1. 全局配置与模型加载
# ==========================================

# 路径配置 (请确保路径正确)
BASE_MODEL_PATH = './checkpoints/weights/noob_vpred_1.0/NoobAI-XL-Vpred-v1.0.safetensors'
CONTROLNET_PATH = './checkpoints/weights/noob_openpose/openpose_pre.safetensors'
TAG_STYLE_PATH  = 'checkpoints/tags/style_transfer_shuffle.json'

# Prompt 配置
BACKGROUD_PROMPT = 'simple_background, white_background' # 移除过多括号，保持干净
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# 加载 Tag 分类配置
print(">> [系统启动] 正在加载 Tag 配置...")
with open(TAG_STYLE_PATH, 'r', encoding='utf-8') as f:
    TAG_CATEGORIES = json.load(f)

# 加载模型
print(">> [系统启动] 正在加载 ControlNet...")
controlnet = ControlNetModel.from_single_file(
    CONTROLNET_PATH, torch_dtype=torch.float16, use_safetensors=True
)

print(">> [系统启动] 正在加载 NoobAI-XL...")
pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    BASE_MODEL_PATH,
    controlnet=controlnet, 
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config,
    prediction_type="v_prediction", 
    rescale_betas_zero_snr=True
)

pipe.to('cuda')
# 如果显存紧张 (>12GB <16GB)，取消下面这行的注释
# pipe.enable_vae_tiling()
print(">> [系统启动] 模型加载完毕。")

# ==========================================
# 2. 核心功能函数
# ==========================================

def build_text_prompt(character_prompt, raw_tags_str):
    """
    构建清洗后的 Prompt，移除 None 和多余逗号
    """
    # 1. 解析原始 Tags
    # tag_classify 返回如: {'cloth': 'school uniform', 'action': 'standing', ...} 
    # 或者 {'cloth': 'None', ...}
    processed_tags = tag_classify(raw_tags_str, TAG_CATEGORIES)
    
    components = []
    
    # 2. 添加固定前缀
    components.append(BACKGROUD_PROMPT)
    components.append(character_prompt) # 目标角色
    
    # 3. 动态提取原图特征 (仅提取有效值)
    # 我们通常只需要 cloth 和 action，head 可能会污染目标角色的发色，根据需要决定是否添加
    for key in ['head', 'cloth', 'action']: 
        val = processed_tags.get(key)
        # 核心修复：检查 val 是否存在，是否为 "None"，是否为空字符串
        if val and val != "None" and val.strip():
            components.append(val)
            
    # 4. 添加质量词
    components.append('masterpiece, best quality, anime style')
    
    # 5. 拼接 (去除空项，用逗号空格连接)
    final_prompt = ", ".join([c.strip() for c in components if c])
    
    return final_prompt

def generate_picture(character_base_prompt, raw_tags, pose_data_dict):
    """
    Args:
        character_base_prompt: 角色触发词
        raw_tags: 原始图片 tag 字符串
        pose_data_dict: 内存中的 openpose 字典数据 (不是文件路径)
    """
    
    # 1. 绘制并缩放骨架 (返回 PIL Image)
    pose_image = draw_pose_from_data(pose_data_dict)
    width, height = pose_image.size
    
    # 2. 构建 Prompt
    text_prompt = build_text_prompt(character_base_prompt, raw_tags)
    
    # 打印调试 (只打印前50个字符避免刷屏)
    # print(f"   [Prompt] {text_prompt[:100]}... | Size: {width}x{height}")
    
    # 3. 生成
    result = pipe(
        prompt=text_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=pose_image,
        controlnet_conditioning_scale=1.0, 
        width=width,  # 显式传递修正后的尺寸
        height=height,
        num_inference_steps=28,
        guidance_scale=7.0,
        output_type="pil"
    ).images[0]
    
    return result, text_prompt # 返回 prompt 以便保存

# ==========================================
# 3. 主流程
# ==========================================

def main():
    # --- 数据加载 ---
    print(">> [加载数据] 读取角色配置...")
    with open('./checkpoints/tags/noobai_characters.json', 'r', encoding='utf-8') as f:
        character_json = json.load(f)

    print(">> [加载数据] 读取 Tags...")
    with open('./data/exported_posts.json', 'r', encoding='utf-8') as f:
        tags_json = json.load(f)

    print(">> [加载数据] 读取 OpenPose...")
    openpose_list = []
    with open('./data/openpose_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    openpose_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # --- 映射构建 ---
    id_tag_mapping = {}
    for item in tags_json:
        if isinstance(item, dict) and 'id' in item and 'tag_string' in item:
            id_tag_mapping[str(item['id'])] = item['tag_string']

    # --- 筛选有效数据 ---
    valid_candidates = []
    for pose_data in openpose_list:
        pid = str(pose_data.get("pid"))
        people = pose_data.get("people")
        
        # 必须有人且有 Tags
        if people and pid in id_tag_mapping:
            valid_candidates.append({
                "pid": pid,
                "pose_data": pose_data,
                "raw_tags": id_tag_mapping[pid]
            })

    total_valid = len(valid_candidates)
    print(f">> [数据预处理] 有效匹配数据: {total_valid} 条")
    if total_valid == 0: return

    # --- 任务循环 ---
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    for char_idx, char_info in enumerate(character_json):
        char_name = char_info.get("name", "unknown")
        char_trigger = char_info.get("tags", "") 
        safe_char_name = "".join([c if c.isalnum() else "_" for c in char_name])

        # 随机抽取任务
        target_count = random.randint(100, 1000)
        actual_count = min(target_count, total_valid)
        selected_tasks = random.sample(valid_candidates, actual_count)

        print(f"\n>> [生成中] 角色: {char_name} | 目标: {actual_count} 张")

        for i, task in enumerate(selected_tasks):
            pid = task['pid']
            pose_data = task['pose_data'] # 直接拿字典
            raw_tags = task['raw_tags']

            base_filename = f"{pid}_{safe_char_name}"
            image_save_path = os.path.join(output_dir, f"{base_filename}.png")
            txt_save_path = os.path.join(output_dir, f"{base_filename}.txt")

            if os.path.exists(image_save_path):
                continue

            try:
                # 调用生成 (直接传字典)
                image, used_prompt = generate_picture(
                    character_base_prompt=char_trigger,
                    raw_tags=raw_tags,
                    pose_data_dict=pose_data 
                )

                image.save(image_save_path)
                
                # 保存实际使用的 Prompt
                with open(txt_save_path, 'w', encoding='utf-8') as tf:
                    tf.write(used_prompt)

                if (i + 1) % 10 == 0:
                    print(f"   -> {i+1}/{actual_count} 完成 (PID: {pid})")

            except Exception as e:
                print(f"   !! [错误] PID: {pid} - {e}")
                torch.cuda.empty_cache() # 释放显存防止 OOM

    print("\n>> [Done] 所有任务完成。")

if __name__ == "__main__":
    main()