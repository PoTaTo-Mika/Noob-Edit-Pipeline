import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from PIL import Image
from main_pipeline.character_transfer.tag_classify import tag_classify
import json
from tools.pose_draw import draw_pose_from_json

# ==========================================
#  全局模型加载 (只运行一次)
# ==========================================

# 路径配置
base_model_path = './checkpoints/weights/noob_vpred_1.0/noobai-v-pred-1.0.safetensors'
controlnet_path = './checkpoints/weights/noob_openpose/openpose_pre.safetensors'

# 通用负面提示词 
DEFAULT_NEGATIVE_PROMPT = None
BACKGROUD_PROMPT = '(((simple_background, white_background))),'

with open('checkpoints/tags/style_transfer_shuffle.json') as f:
        TAG_CATEGORIES = json.load(f)

print(">> [系统启动] 正在加载 ControlNet...")
controlnet = ControlNetModel.from_single_file(
    controlnet_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)

print(">> [系统启动] 正在加载主模型 NoobAI-XL...")
pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    base_model_path,
    controlnet=controlnet, 
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# 配置 V-Prediction 调度器
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config,
    prediction_type="v_prediction", 
    rescale_betas_zero_snr=True
)

# 移动到 GPU
print(">> [系统启动] 正在移动模型至 CUDA...")
pipe.to('cuda')
print(">> [系统启动] 模型加载完毕，等待任务...")

# ==========================================
# 2. 推理函数定义
# ==========================================

# 我们数据就准备成openpose的json+随机抽取text_prompt的形式，每个json读取之后build几个角色
def build_text_prompt(character_prompt, raw_prompt):
    # 对于原始的prompt，我们要用函数提取里面的内容
    processed_prompt = tag_classify(raw_prompt,TAG_CATEGORIES) # 这里的processed_prompt是json
    cloth = processed_prompt.get("cloth", "None")
    action = processed_prompt.get("action", "None") 
    head = processed_prompt.get("head", "None")
    # 顺序不能变
    final_prompt = BACKGROUD_PROMPT + head +','+ character_prompt +','+ cloth +','+ action + 'masterpiece,best quality'
    return final_prompt 

def generate_picture(character_base_prompt, raw_tags, openpose_json_path):
    """
    Args:
        character_base_prompt: 角色基础词，如 "dusk_(arknights), 1girl"
        raw_tags: 原始 tag 串，用于提取动作和服装
        openpose_json_path: 骨架 JSON 文件路径
    """
    
    # 1. 绘制骨架图
    pose_image = draw_pose_from_json(openpose_json_path)
    width, height = pose_image.size
    
    # 2. 构建 Prompt
    text_prompt = build_text_prompt(character_base_prompt, raw_tags)
    print(f"Generate Prompt: {text_prompt}")
    
    # 3. 生成
    result = pipe(
        prompt=text_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        image=pose_image,
        controlnet_conditioning_scale=1.0, 
        width=width,
        height=height,
        num_inference_steps=28,
        guidance_scale=7.0,
        output_type="pil"
    ).images[0]
    
    return result

def main():
     pass