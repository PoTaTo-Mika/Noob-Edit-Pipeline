import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from PIL import Image
from main_pipeline.character_transfer.tag_classify import tag_classify
import json

# ==========================================
#  全局模型加载 (只运行一次)
# ==========================================

# 路径配置
base_model_path = './noobai-v-pred-1.0.safetensors'
controlnet_path = './noobai-xl-controlnet-openpose.safetensors'

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
    processed_prompt = tag_classify(raw_prompt) # 这里的processed_prompt是json
    cloth = processed_prompt.get("cloth", "None")
    action = processed_prompt.get("action", "None") 
    head = processed_prompt.get("head", "None")
    # 顺序不能变
    final_prompt = BACKGROUD_PROMPT + head +','+ character_prompt +','+ cloth +','+ action
    return final_prompt 

def generate_picture(text_prompt, openpose_img):
    
    # 自动获取 OpenPose 图片的尺寸，确保生成图与骨架严丝合缝
    width, height = openpose_img.size
    # 执行生成
    result = pipe(
        prompt=text_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        image=openpose_img,              # 传入骨架图
        controlnet_conditioning_scale=1.0, # 严格听从骨架
        width=width,
        height=height,
        num_inference_steps=28,          # NoobAI 标准步数
        guidance_scale=7.0,              # 二次元标准 CFG
        output_type="pil"
    ).images[0]
    
    return result

# ==========================================
# 3. 简单测试调用 (示例)
# ==========================================
if __name__ == "__main__":
    # 模拟数据 Pipeline 传入的数据
    test_prompt = 'dusk_(arknights), masterpiece, best quality'
    
    # 创建一个测试用的 dummy pose (实际场景中你会从上游传入真实的 openpose image)
    # 假设上游已经把图片处理成了 SDXL 喜欢的 1024x1024
    test_pose_img = Image.new("RGB", (1024, 1024), (0, 0, 0))
    
    print(f"正在生成: {test_prompt} ...")
    
    # 调用函数
    generated_image = generate_picture(test_prompt, test_pose_img)
    
    # 保存结果
    generated_image.save("result_test.png")
    print("测试生成完成。")